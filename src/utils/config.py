"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.

Adapted from opensource project GraphCMR (https://github.com/nkolot/GraphCMR/) and Pose2Mesh (https://github.com/hongsukchoi/Pose2Mesh_RELEASE)

"""

from yacs.config import CfgNode as CN

folder_path = 'src/modeling/'
JOINT_REGRESSOR_TRAIN_EXTRA = folder_path + 'data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M_correct = folder_path + 'data/J_regressor_h36m_correct.npy'
SMPL_FILE = folder_path + 'data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
SMPL_Male = folder_path + 'data/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
SMPL_Female = folder_path + 'data/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
JOINTS_IDX = [8, 5, 29, 30, 4, 7, 21, 19, 17, 16, 18, 20, 31, 32, 33, 34, 35, 36, 37, 24, 26, 25, 28, 27]


"""
We follow the body joint definition, loss functions, and evaluation metrics from 
open source project GraphCMR (https://github.com/nkolot/GraphCMR/)

Each dataset uses different sets of joints.
We use a superset of 24 joints such that we include all joints from every dataset.
If a dataset doesn't provide annotations for a specific joint, we simply ignore it.
The joints used here are:
"""
J24_NAME = ('R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder',
'L_Elbow','L_Wrist','Neck','Top_of_Head','Pelvis','Thorax','Spine','Jaw','Head','Nose','L_Eye','R_Eye','L_Ear','R_Ear')
H36M_J17_NAME = ( 'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head',
                  'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
J24_TO_J14 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
H36M_J17_TO_J14 = [3, 2, 1, 4, 5, 6, 16, 15, 14, 11, 12, 13, 8, 10]


##### CONFIGS #####
hparams = CN()
hparams.DATASET = '3dpw'
# model hparams
hparams.MODEL = CN()
hparams.MODEL.BACKBONE = 'resnet50' # hrnet_w32-conv, hrnet_w32-interp
hparams.MODEL.NUM_JOINTS = 24
hparams.MODEL.SOFTMAX_TEMP = 1.
hparams.MODEL.NUM_FEATURES_SMPL = 64
hparams.MODEL.NUM_DECONV_LAYERS = 3
hparams.MODEL.NUM_DECONV_FILTERS = 256
hparams.MODEL.POSE_MLP_NUM_LAYERS = 1
hparams.MODEL.SHAPE_MLP_NUM_LAYERS = 1
hparams.MODEL.POSE_MLP_HIDDEN_SIZE = 256
hparams.MODEL.SHAPE_MLP_HIDDEN_SIZE = 256
hparams.MODEL.PATCH_RES = 9
hparams.MODEL.TNT_DEPTH = 2
hparams.MODEL.PRETRAINED_PARE = True
hparams.MODEL.POSITION = False
hparams.MODEL.USE_HEATMAPS = 'part_segm'

def get_hparams_defaults():
    """Get a yacs hparamsNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return hparams.clone()

def update_hparams(hparams_file):
    hparams = get_hparams_defaults()
    hparams.merge_from_file(hparams_file)
    return hparams.clone()

def update_hparams_from_dict(cfg_dict):
    hparams = get_hparams_defaults()
    cfg = hparams.load_cfg(str(cfg_dict))
    hparams.merge_from_other_cfg(cfg)
    return hparams.clone()