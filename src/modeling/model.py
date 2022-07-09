from pickle import NONE
import torch
import torch.nn as nn
from src.modeling.head.tnt_head import TNTHead
from .backbone.utils import get_backbone_info
from .backbone.hrnet import hrnet_w32, hrnet_w64
from .backbone.pose_resnet import get_resnet50
from collections import OrderedDict

def load_pretrained_model(model, state_dict, strict=False, overwrite_shape_mismatch=True, remove_lightning=False):
    if remove_lightning:
        pretrained_keys = state_dict.keys()
        new_state_dict = OrderedDict()
        for pk in pretrained_keys:
            if pk.startswith('model.'):
                new_state_dict[pk.replace('model.', '')] = state_dict[pk]
            else:
                new_state_dict[pk] = state_dict[pk]

        model.load_state_dict(new_state_dict, strict=strict)
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        if overwrite_shape_mismatch:
            model_state_dict = model.state_dict()
            pretrained_keys = state_dict.keys()
            model_keys = model_state_dict.keys()

            updated_pretrained_state_dict = state_dict.copy()

            for pk in pretrained_keys:
                if pk in model_keys:
                    if model_state_dict[pk].shape != state_dict[pk].shape:
            

                        if pk == 'model.head.fc1.weight':
                            updated_pretrained_state_dict[pk] = torch.cat(
                                [state_dict[pk], state_dict[pk][:,-7:]], dim=-1
                            )
                            continue
                        else:
                            del updated_pretrained_state_dict[pk]

            model.load_state_dict(updated_pretrained_state_dict, strict=False)
        else:
            raise RuntimeError('there are shape inconsistencies between pretrained ckpt and current ckpt')
    return model

class Model(nn.Module):
    def __init__(
            self,
            num_joints=24,
            softmax_temp=1.0,
            num_features_smpl=64,
            backbone='resnet50',
            pose_mlp_num_layers=1,
            shape_mlp_num_layers=1,
            pose_mlp_hidden_size=256,
            shape_mlp_hidden_size=256,
            use_heatmaps='',
            num_deconv_layers=3,
            num_deconv_filters=256,
            init_xavier=False,
            seq=1,
            patch_res=9,
            depth=2,
            position=False
    ):
        super(Model, self).__init__()
        if backbone is 'hrnet_resnet50':
            self.backbone = get_resnet50(pretrained='data/pretrained_models/pose_coco/pose_resnet_50_256x192.pth')
        else:
            backbone, use_conv = backbone.split('-')
            self.backbone = eval(backbone)(
                pretrained=True,
                downsample=False,
                use_conv=(use_conv == 'conv')
            )     
        self.head = TNTHead(
            num_joints=num_joints,
            num_input_features=get_backbone_info(backbone)['n_output_channels'],
            softmax_temp=softmax_temp,
            num_deconv_layers=num_deconv_layers,
            num_deconv_filters=[num_deconv_filters] * num_deconv_layers,
            num_features_smpl=num_features_smpl,
            final_conv_kernel=1,
            pose_mlp_num_layers=pose_mlp_num_layers,
            shape_mlp_num_layers=shape_mlp_num_layers,
            pose_mlp_hidden_size=pose_mlp_hidden_size,
            shape_mlp_hidden_size=shape_mlp_hidden_size,
            use_heatmaps=use_heatmaps,
            backbone=backbone,
            init_xavier=init_xavier,
            seq=seq,
            patch_res=patch_res,
            depth=depth,
            position=position
        )
        
    def forward(
            self,
            images,
            gt_segm=None,
            return_heat=False
    ):
        batch = images.shape[0]
        if len(images.shape) > 4:
            seq = images.shape[1]
            images = images.view(batch*seq, *images.shape[2:])
        else:
            seq = 1
        features = self.backbone(images)
        if return_heat:
            smpl_output, pred_heats = self.head(features, gt_segm=gt_segm, return_heat=True)
        else:
            smpl_output = self.head(features, gt_segm=gt_segm, return_heat=False)
        
        if return_heat:
            return smpl_output['pred_cam'], smpl_output['pred_pose'], smpl_output['pred_shape'], pred_heats
        else:
            return smpl_output['pred_cam'], smpl_output['pred_pose'], smpl_output['pred_shape'], None

def get_model(model_cfg):
    PARE_CKPT = 'data/pretrained_models/pare_checkpoint.ckpt'
    model = Model(
        backbone=model_cfg.MODEL.BACKBONE,
        num_joints=model_cfg.MODEL.NUM_JOINTS,
        softmax_temp=model_cfg.MODEL.SOFTMAX_TEMP,
        num_features_smpl=model_cfg.MODEL.NUM_FEATURES_SMPL,
        pose_mlp_num_layers=model_cfg.MODEL.POSE_MLP_NUM_LAYERS,
        shape_mlp_num_layers=model_cfg.MODEL.SHAPE_MLP_NUM_LAYERS,
        pose_mlp_hidden_size=model_cfg.MODEL.POSE_MLP_HIDDEN_SIZE,
        shape_mlp_hidden_size=model_cfg.MODEL.SHAPE_MLP_HIDDEN_SIZE,
        use_heatmaps=model_cfg.MODEL.USE_HEATMAPS,
        num_deconv_layers=model_cfg.MODEL.NUM_DECONV_LAYERS,
        num_deconv_filters=model_cfg.MODEL.NUM_DECONV_FILTERS,
        seq=1,
        patch_res=model_cfg.MODEL.PATCH_RES,
        depth=model_cfg.MODEL.TNT_DEPTH,
        init_xavier=True,
        position=model_cfg.MODEL.POSITION
    )
    if model_cfg.MODEL.PRETRAINED_PARE and model_cfg.MODEL.BACKBONE == 'hrnet_w32-conv':
        model.eval()
        states = torch.load(PARE_CKPT, map_location='cpu')['state_dict']
        for k, v in states.items():
            states[k] = v.cpu()
        load_pretrained_model(model, states, overwrite_shape_mismatch=True, remove_lightning=True)

    return model
