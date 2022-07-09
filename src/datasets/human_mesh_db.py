import joblib
import numpy as np
import torch
import torchvision.transforms as transforms
from src.utils.tsv_file_ops import load_linelist_file, load_from_yaml_file, find_file_path_in_yaml
from src.utils.image_ops import crop, flip_img, flip_pose, flip_kp, transform, rot_aa
import cv2
from skimage.util.shape import view_as_windows

def split_into_chunks(vid_names, seqlen, stride):
    video_start_end_indices = []

    video_names, group = np.unique(vid_names, return_index=True)
    perm = np.argsort(group)
    video_names, group = video_names[perm], group[perm]

    indices = np.split(np.arange(0, vid_names.shape[0]), group[1:])

    for idx in range(len(video_names)):
        indexes = indices[idx]
        if indexes.shape[0] < seqlen:
            continue
        chunks = view_as_windows(indexes, (seqlen,), step=stride)
        start_finish = chunks[:, (0, -1)].tolist()
        video_start_end_indices += start_finish

    return video_start_end_indices

class MeshDBDataset(object):
    def __init__(self, db_file, seq=1, stride=[1], is_train=True):
        super().__init__()
        self.db_file = db_file
        self.db = [joblib.load(x) for x in self.db_file]
        self.seq = seq

        vid_indices = []
        for i in range(len(self.db)):
            vid_indices.append(split_into_chunks(self.db[i]['vid_name'], seq, stride[i]))

        self.vid_indices = [(i, vid_indices[i][j]) for i in range(len(self.db_file)) for j in range(len(vid_indices[i]))]
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.is_train = is_train
        self.scale_factor = 0.25 # rescale bounding boxes by a factor of [1-options.scale_factor,1+options.scale_factor]
        self.noise_factor = 0.4
        self.rot_factor = 30 # Random rotation in the range [-rot_factor, rot_factor]
        self.img_res = 224
        self.joints_definition = ('R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder',
        'L_Elbow','L_Wrist','Neck','Head')
        self.lhip_index = self.joints_definition.index('L_Hip')
        self.rhip_index = self.joints_definition.index('R_Hip')
    
    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1.           # scaling
        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
	    
            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.noise_factor, 1+self.noise_factor, 3)
	    
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.rot_factor,
                    max(-2*self.rot_factor, np.random.randn()*self.rot_factor))
	    
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.scale_factor,
                    max(1-self.scale_factor, np.random.randn()*self.scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0
	
        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        # print(scale)
        rgb_img = crop(rgb_img, center, scale, 
                      [self.img_res, self.img_res], rot=rot)
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)

        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
                                  [self.img_res, self.img_res], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/self.img_res - 1.
        # flip the x coordinates
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1]) 
        # flip the x coordinates
        if f:
            S = flip_kp(S)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose = pose.astype('float32')
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose
    
    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, idx):
        indices = self.vid_indices[idx]
        start = indices[1][0]
        end = indices[1][1]
        # print(idx, self.db[indices[0]]['img_name'][start], self.db[indices[0]]['bbox'][start])
        img = [cv2.cvtColor(cv2.imread(self.db[indices[0]]['img_name'][i]), cv2.COLOR_BGR2RGB) for i in range(start, end+1)]
        center = np.asarray([self.db[indices[0]]['bbox'][i].copy()[:2] for i in range(start, end+1)])
        scale = np.asarray([self.db[indices[0]]['bbox'][i].copy()[2:].max() / 200. for i in range(start, end+1)])
        
        has_2d_joints = np.asarray([self.db[indices[0]]['has_2d_joints'][i] for i in range(start, end+1)])
        has_3d_joints = np.asarray([self.db[indices[0]]['has_3d_joints'][i] for i in range(start, end+1)])
        joints_2d = np.asarray([self.db[indices[0]]['2d_joints'][i] for i in range(start, end+1)])
        joints_3d = np.asarray([self.db[indices[0]]['3d_joints'][i] for i in range(start, end+1)])

        # Get SMPL parameters, if available
        has_smpl = np.asarray([self.db[indices[0]]['has_smpl'][i] for i in range(start, end+1)])
        pose = np.asarray([self.db[indices[0]]['pose'][i] for i in range(start, end+1)])
        betas = np.asarray([self.db[indices[0]]['betas'][i] for i in range(start, end+1)])

        try:
            gender = self.db[indices[0]]['gender'][start]
        except KeyError:
            gender = 'none'

        # Get augmentation parameters
        flip,pn,rot,sc = self.augm_params()

        # Process image
        # print(scale)
        img = [self.rgb_processing(img[i], center[i], sc*scale[i], rot, flip, pn) for i in range(self.seq)]
        img = [torch.from_numpy(img[i]).float() for i in range(self.seq)]
        # Store image before normalization to use it in visualization
        transfromed_img = [self.normalize_img(img[i]) for i in range(self.seq)]

        # normalize 3d pose by aligning the pelvis as the root (at origin)
        # root_pelvis = joints_3d[:, self.pelvis_index,:-1]
        root_pelvis = (joints_3d[:, self.lhip_index,:-1] + joints_3d[:, self.rhip_index,:-1]) / 2.
        joints_3d[:,:,:-1] = joints_3d[:,:,:-1] - root_pelvis[:,None,:]
        # 3d pose augmentation (random flip + rotation, consistent to image and SMPL)
        joints_3d_transformed = np.asarray([self.j3d_processing(joints_3d.copy()[i], rot, flip) for i in range(self.seq)])
        # 2d pose augmentation
        joints_2d_transformed = np.asarray([self.j2d_processing(joints_2d.copy()[i], center[i], sc*scale[i], rot, flip) for i in range(self.seq)])

        ###################################
        # Masking percantage
        # We observe that 30% works better for human body mesh. Further details are reported in the paper.
        mvm_percent = 0.3
        msm_percent = 0.6
        # ###################################
        
        mjm_mask = np.ones((14,1))
        msm_mask = np.ones((self.seq, 1))

        if self.is_train:
            pb = np.random.random_sample()
            masked_num = int(pb * msm_percent * self.seq) # at most x% of the seqs could be masked
            indices = np.random.choice(np.arange(self.seq),replace=False,size=masked_num)
            msm_mask[indices,:] = 0.0

        if self.is_train:
            num_joints = 14
            pb = np.random.random_sample()
            masked_num = int(pb * mvm_percent * num_joints) # at most x% of the joints could be masked
            indices = np.random.choice(np.arange(num_joints),replace=False,size=masked_num)
            mjm_mask[indices,:] = 0.0
        # mjm_mask = torch.from_numpy(mjm_mask).float()

        mvm_mask = np.ones((431,1))
        if self.is_train:
            num_vertices = 431
            pb = np.random.random_sample()
            masked_num = int(pb * mvm_percent * num_vertices) # at most x% of the vertices could be masked
            indices = np.random.choice(np.arange(num_vertices),replace=False,size=masked_num)
            mvm_mask[indices,:] = 0.0
        # mvm_mask = torch.from_numpy(mvm_mask).float()

        meta_data = {}
        meta_data['ori_img'] = torch.stack(img, 0)
        meta_data['pose'] = torch.from_numpy(np.asarray([self.pose_processing(pose[i], rot, flip) for i in range(self.seq)])).float()
        meta_data['betas'] = torch.from_numpy(betas).float()
        meta_data['joints_3d'] = torch.from_numpy(joints_3d_transformed).float()
        meta_data['has_3d_joints'] = torch.from_numpy(has_3d_joints).float()
        meta_data['has_smpl'] = torch.from_numpy(has_smpl).float()

        meta_data['mjm_mask'] = torch.from_numpy(mjm_mask).float()
        meta_data['mvm_mask'] = torch.from_numpy(mvm_mask).float()
        meta_data['msm_mask'] = torch.from_numpy(msm_mask).float()
        # Get 2D keypoints and apply augmentation transforms
        meta_data['has_2d_joints'] = torch.from_numpy(has_2d_joints).float()
        meta_data['joints_2d'] = torch.from_numpy(joints_2d_transformed).float()
        meta_data['scale'] = torch.from_numpy(sc * scale).float()
        meta_data['center'] = torch.from_numpy(center).float()
        meta_data['gender'] = gender
        return 'none', torch.stack(transfromed_img, 0), meta_data





class MeshDBYamlDataset(MeshDBDataset):
    """ DBDataset taking a Yaml file for easy function call
    """
    def __init__(self, yaml_file, is_train=True):
        self.cfg = load_from_yaml_file(yaml_file)
        db_file = self.cfg.get('db_file', [])
        if type(db_file) is str:
            db_file = [db_file]
        seq = self.cfg.get('seq', 1)
        stride = self.cfg.get('stride', [1] * len(db_file))
        if not isinstance(stride, list):
            stride = [stride] * len(db_file)

        super(MeshDBYamlDataset, self).__init__(
            db_file, seq, stride, is_train)
