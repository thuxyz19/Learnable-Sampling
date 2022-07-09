# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ...utils.geometry import rot6d_to_rotmat
from ..layers.softargmax import softargmax2d, get_heatmap_preds
from ..layers import LocallyConnected2d
from functools import partial

BN_MOMENTUM = 0.1

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class RelativeAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., seq=9):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.B = nn.Parameter(torch.randn(1, num_heads, seq, seq), requires_grad=True)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        bias = self.B
        attn = (q @ k.transpose(-2, -1)) * self.scale + bias
        if mask is not None:
            mask = mask.view(B, 1, N, N)
            attn = mask + attn
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., factor=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask.view(B, 1, N, N)
            attn = mask + attn


        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_type=Attention, factor_q=1, factor_k=3, seq=9):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_type = attn_type
        if attn_type is Attention:
            self.attn = attn_type(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        else:
            self.attn = attn_type(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, seq=seq)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        if self.attn_type is RelativeAttention:
            x = x + self.drop_path(self.attn(self.norm1(x), mask))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TNT(nn.Module):
    def __init__(self, indim, seq, num_joints=24, patch_res=9, res=56, depth=4, s_conv=False, position=False):
        super().__init__()
        self.res = res
        self.patch_res = patch_res
        self.depth = depth
        self.seq = seq
        self.num_joints = num_joints
        self.indim = indim
        self.position = position
        self.down_sample = nn.AdaptiveAvgPool2d((patch_res, patch_res))
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.inner_blocks = nn.ModuleList([
            Block(indim, num_heads=8, mlp_ratio=2.0, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, attn_type=RelativeAttention, seq=patch_res*patch_res)
            for _ in range(depth)
        ])
        if self.position:
            self.coordinates = nn.Parameter(torch.randn(indim, 56, 56), requires_grad=True)
        self.s_conv = s_conv
        assert self.seq == 1, "Have not implemented for sequences yet"
        self.outer_s_blocks = nn.ModuleList([
            Block(indim, num_heads=8, mlp_ratio=2.0, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, attn_type=RelativeAttention, seq=num_joints)
            for _ in range(depth)
        ])
        self.pixel_down_0 = nn.Sequential(
                nn.Linear(indim, indim//4),
                nn.GELU(),
                nn.Linear(indim//4, indim//4)
            )
        self.pixel2patch_0 = nn.Linear(self.patch_res*self.patch_res*indim//4, indim)
        self.pixel_down = nn.ModuleList([
            nn.Sequential(
                nn.Linear(indim, indim//4),
                nn.GELU(),
                nn.Linear(indim//4, indim//4)
            )
            for _ in range(depth)
        ])
        self.pixel2patch = nn.ModuleList([
            nn.Linear(self.patch_res*self.patch_res*indim//4, indim)
            for _ in range(depth)
        ])
        
    
    def sample_conv(self, x, heat):
        B, C = x.shape[:2]
        J = self.num_joints
        heat = heat.view(B, J+J*self.patch_res*self.patch_res, self.res*self.res)
        heat = F.softmax(heat, dim=-1)
        heat_weight = torch.ones_like(heat[:, :J, ...], device=heat.device, dtype=heat.dtype)
        if self.position:
            coordinates = self.coordinates.view(1, C, self.res*self.res).repeat(B, 1, 1)
        heat_sample = heat[:, J:, ...].view(B, J, self.patch_res*self.patch_res, self.res*self.res)
        x = x.view(B ,C, -1)
        x_patch = []
        for i in range(self.num_joints):
            x_i = x * heat_weight[:, None, i]
            x_i = torch.einsum('bcn,bnm->bcm', x_i, heat_sample[:, i, ...].permute(0, 2, 1).contiguous())
            if self.position:
                position =  torch.einsum('bcn,bnm->bcm', coordinates, heat_sample[:, i, ...].permute(0, 2, 1).contiguous())
                x_i = x_i + position
            x_patch.append(x_i)
        x_patch = torch.stack(x_patch, 2)
        x_patch = x_patch.view(B, C, J, self.patch_res, self.patch_res)
        x_patch = x_patch.permute(0, 2, 1, 3, 4).contiguous()
        x_patch = x_patch.view(B//self.seq, self.seq, J, C, self.patch_res, self.patch_res)
        return x_patch
        
    def forward(self, features, heatmaps):
        # x: (B*seq, C, H, W) heatmaps: (B*seq, J+self.patch_res*self.patch_res*24, H, W)
        BS, C = features.shape[:2]
        B = BS // self.seq
        pixel_tokens = self.sample_conv(features, heatmaps) # (B, seq, J, C, h, w)
        pixel_tokens = pixel_tokens.view(B*self.seq*self.num_joints, C, self.patch_res*self.patch_res).permute(0, 2, 1) # (B*seq*J, h*w, C)
        patch_tokens = self.pixel2patch_0(self.pixel_down_0(pixel_tokens).view(B*self.seq*self.num_joints, -1))
        for i in range(self.depth):
            pixel_tokens = self.inner_blocks[i](pixel_tokens)
            patch_tokens_add = self.pixel2patch[i](self.pixel_down[i](pixel_tokens).view(B*self.seq*self.num_joints, -1))
            patch_tokens = patch_tokens + patch_tokens_add
            patch_tokens = patch_tokens.view(B*self.seq, self.num_joints, -1)
            patch_tokens = self.outer_s_blocks[i](patch_tokens)
            patch_tokens = patch_tokens.view(B*self.seq*self.num_joints, -1)
        patch_tokens = patch_tokens.view(B, self.seq, self.num_joints, -1)
        patch_tokens = patch_tokens.view(B*self.seq, self.num_joints, -1).permute(0, 2, 1).contiguous()
        return patch_tokens

class TNTHead(nn.Module):
    def __init__(
            self,
            num_joints,
            num_input_features,
            softmax_temp=1.0,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_camera_params=3,
            num_features_smpl=64,
            final_conv_kernel=1,
            pose_mlp_num_layers=1,
            shape_mlp_num_layers=1,
            pose_mlp_hidden_size=256,
            shape_mlp_hidden_size=256,
            use_heatmaps='',
            backbone='hrnet_w32',
            init_xavier=False,
            seq=1,
            patch_res=9,
            depth=2,
            position=False
    ):
        super(TNTHead, self).__init__()
        self.backbone = backbone
        self.num_joints = num_joints
        self.deconv_with_bias = False
        self.use_heatmaps = use_heatmaps
        self.pose_mlp_num_layers = pose_mlp_num_layers
        self.shape_mlp_num_layers = shape_mlp_num_layers
        self.pose_mlp_hidden_size = pose_mlp_hidden_size
        self.shape_mlp_hidden_size = shape_mlp_hidden_size
        self.num_input_features = num_input_features
        self.seq = seq
        self.patch_res = patch_res
        self.depth = depth
        self.position = position
        self.tnt = TNT(indim=128, seq=self.seq, num_joints=24, patch_res=self.patch_res, res=56, depth=self.depth, position=self.position)
        self.tnt_cam = TNT(indim=num_features_smpl, seq=self.seq, num_joints=24, patch_res=self.patch_res, res=56, depth=self.depth, position=self.position)
        self.heat_sample = nn.Conv2d(in_channels=128, out_channels=self.patch_res*self.patch_res*24, kernel_size=1, stride=1, padding=0)

        self.keypoint_deconv_layers = self._make_conv_layer(
            num_deconv_layers,
            num_deconv_filters,
            (3,)*num_deconv_layers,
        )
        self.num_input_features = num_input_features
        self.smpl_deconv_layers = self._make_conv_layer(
            num_deconv_layers,
            num_deconv_filters,
            (3,)*num_deconv_layers,
        )

        pose_mlp_inp_dim = num_deconv_filters[-1]
        smpl_final_dim = num_features_smpl
        shape_mlp_inp_dim = num_joints * smpl_final_dim
        self.keypoint_final_layer = nn.Conv2d(
            in_channels=num_deconv_filters[-1],
            out_channels=num_joints+1 if self.use_heatmaps in ('part_segm', 'part_segm_pool') else num_joints,
            kernel_size=final_conv_kernel,
            stride=1,
            padding=1 if final_conv_kernel == 3 else 0,
        )
        self.smpl_final_layer = nn.Conv2d(
            in_channels=num_deconv_filters[-1],
            out_channels=smpl_final_dim,
            kernel_size=final_conv_kernel,
            stride=1,
            padding=1 if final_conv_kernel == 3 else 0,
        )

        # temperature for softargmax function
        self.register_buffer('temperature', torch.tensor(softmax_temp))

        self.pose_mlp_inp_dim = pose_mlp_inp_dim
        self.shape_mlp_inp_dim = shape_mlp_inp_dim
        # here we use 2 different MLPs to estimate shape and camera
        # They take a channelwise downsampled version of smpl features
        self.shape_mlp = self._get_shape_mlp(output_size=10)
        self.cam_mlp = self._get_shape_mlp(output_size=num_camera_params)

        # for pose each joint has a separate MLP
        # weights for these MLPs are not shared
        # hence we use Locally Connected layers
        # TODO support kernel_size > 1 to access context of other joints
        self.pose_mlp = self._get_pose_mlp(num_joints=num_joints, output_size=6)

        if init_xavier:
            nn.init.xavier_uniform_(self.shape_mlp.weight, gain=0.01)
            nn.init.xavier_uniform_(self.cam_mlp.weight, gain=0.01)
            nn.init.xavier_uniform_(self.pose_mlp.weight, gain=0.01)

    def _get_shape_mlp(self, output_size):
        if self.shape_mlp_num_layers == 1:
            return nn.Linear(self.shape_mlp_inp_dim, output_size)

        module_list = []
        for i in range(self.shape_mlp_num_layers):
            if i == 0:
                module_list.append(
                    nn.Linear(self.shape_mlp_inp_dim, self.shape_mlp_hidden_size)
                )
            elif i == self.shape_mlp_num_layers - 1:
                module_list.append(
                    nn.Linear(self.shape_mlp_hidden_size, output_size)
                )
            else:
                module_list.append(
                    nn.Linear(self.shape_mlp_hidden_size, self.shape_mlp_hidden_size)
                )
        return nn.Sequential(*module_list)

    def _get_pose_mlp(self, num_joints, output_size):
        if self.pose_mlp_num_layers == 1:
            return LocallyConnected2d(
                in_channels=self.pose_mlp_inp_dim,
                out_channels=output_size,
                output_size=[num_joints, 1],
                kernel_size=1,
                stride=1,
            )

        module_list = []
        for i in range(self.pose_mlp_num_layers):
            if i == 0:
                module_list.append(
                    LocallyConnected2d(
                        in_channels=self.pose_mlp_inp_dim,
                        out_channels=self.pose_mlp_hidden_size,
                        output_size=[num_joints, 1],
                        kernel_size=1,
                        stride=1,
                    )
                )
            elif i == self.pose_mlp_num_layers - 1:
                module_list.append(
                    LocallyConnected2d(
                        in_channels=self.pose_mlp_hidden_size,
                        out_channels=output_size,
                        output_size=[num_joints, 1],
                        kernel_size=1,
                        stride=1,
                    )
                )
            else:
                module_list.append(
                    LocallyConnected2d(
                        in_channels=self.pose_mlp_hidden_size,
                        out_channels=self.pose_mlp_hidden_size,
                        output_size=[num_joints, 1],
                        kernel_size=1,
                        stride=1,
                    )
                )
        return nn.Sequential(*module_list)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_conv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_conv_layers is different len(num_conv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_conv_layers is different len(num_conv_filters)'
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.Conv2d(
                    in_channels=self.num_input_features,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=1,
                    padding=padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.num_input_features = planes

        return nn.Sequential(*layers)


    def forward(self, features, gt_segm=None, return_heat=False):
        batch_size = features.shape[0]
        output = {}

        ############## 2D PART BRANCH FEATURES ##############
        part_feats = self._get_2d_branch_feats(features)

        ############## GET PART ATTENTION MAP ##############
        part_attention = self._get_part_attention_map(part_feats, output)
        part_attention_sample = self.heat_sample(part_feats)
        part_attention = torch.cat([part_attention, part_attention_sample], 1)
        heat_sample = part_attention_sample.view(batch_size, self.num_joints, self.patch_res*self.patch_res, 56, 56).mean(2)
        pred_kp2d, _ = softargmax2d(heat_sample, self.temperature) # get_heatmap_preds(heatmaps)
        output['pred_kp2d'] = pred_kp2d

        ############## 3D SMPL BRANCH FEATURES ##############
        smpl_feats = self._get_3d_smpl_feats(features)

        ############## SAMPLE LOCAL FEATURES ##############
        if gt_segm is not None:
            gt_segm = F.interpolate(gt_segm.unsqueeze(1).float(), scale_factor=(1/4, 1/4), mode='nearest').long().squeeze(1)
            part_attention = F.one_hot(gt_segm.to('cpu'), num_classes=self.num_joints + 1).permute(0,3,1,2).float()[:,1:,:,:]
            part_attention = part_attention.to('cuda')
        point_local_feat, cam_shape_feats = self._get_local_feats(smpl_feats, part_attention, output) # point_local_feat: (B, C, J)
        ############## GET FINAL PREDICTIONS ##############
        pred_pose, pred_shape, pred_cam = self._get_final_preds(
            point_local_feat, cam_shape_feats)

        pred_rotmat = rot6d_to_rotmat(pred_pose).reshape(batch_size, 24, 3, 3)
        output.update({
            'pred_pose': pred_rotmat,
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
        })
        if return_heat:
            heat_sample = part_attention_sample.view(batch_size, self.num_joints, self.patch_res*self.patch_res, 56*56)
            heat_sample = F.softmax(heat_sample, dim=-1)
            heat_sample = heat_sample.view(batch_size, self.num_joints, self.patch_res*self.patch_res, 56, 56)
            return output, heat_sample
        else:
            return output

    def _get_local_feats(self, smpl_feats, part_attention, output):
        cam_shape_feats = self.smpl_final_layer(smpl_feats)
        point_local_feat = self.tnt(smpl_feats, part_attention)
        cam_shape_feats = self.tnt_cam(cam_shape_feats, part_attention)
        return point_local_feat, cam_shape_feats

    def _get_2d_branch_feats(self, features):
        part_feats = self.keypoint_deconv_layers(features)
        return part_feats

    def _get_3d_smpl_feats(self, features):
        smpl_feats = self.smpl_deconv_layers(features)
        return smpl_feats

    def _get_part_attention_map(self, part_feats, output):
        heatmaps = self.keypoint_final_layer(part_feats)

        if self.use_heatmaps == 'hm':
            # returns coords between [-1,1]
            pred_kp2d, confidence = get_heatmap_preds(heatmaps)
            output['pred_kp2d'] = pred_kp2d
            output['pred_kp2d_conf'] = confidence
            output['pred_heatmaps_2d'] = heatmaps
        elif self.use_heatmaps == 'hm_soft':
            pred_kp2d, _ = softargmax2d(heatmaps, self.temperature)
            output['pred_kp2d'] = pred_kp2d
            output['pred_heatmaps_2d'] = heatmaps
        elif self.use_heatmaps == 'part_segm':
            output['pred_segm_mask'] = heatmaps
            heatmaps = heatmaps[:,1:,:,:] # remove the first channel which encodes the background
            pred_kp2d, _ = softargmax2d(heatmaps, self.temperature) # get_heatmap_preds(heatmaps)
            output['pred_kp2d'] = pred_kp2d
        elif self.use_heatmaps == 'part_segm_pool':
            output['pred_segm_mask'] = heatmaps
            heatmaps = heatmaps[:,1:,:,:] # remove the first channel which encodes the background
            pred_kp2d, _ = softargmax2d(heatmaps, self.temperature) # get_heatmap_preds(heatmaps)
            output['pred_kp2d'] = pred_kp2d

            for k, v in output.items():
                if torch.any(torch.isnan(v)):
                    print(f'{k} is Nan!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                if torch.any(torch.isinf(v)):
                    print(f'{k} is Inf!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        elif self.use_heatmaps == 'attention':
            output['pred_attention'] = heatmaps
        else:
            # returns coords between [-1,1]
            pred_kp2d, _ = softargmax2d(heatmaps, self.temperature)
            output['pred_kp2d'] = pred_kp2d
            output['pred_heatmaps_2d'] = heatmaps
        return heatmaps

    def _get_final_preds(self, pose_feats, cam_shape_feats):
        return self._pare_get_final_preds(pose_feats, cam_shape_feats)

    def _pare_get_final_preds(self, pose_feats, cam_shape_feats):
        pose_feats = pose_feats.unsqueeze(-1)  #
        shape_feats = cam_shape_feats
        shape_feats = torch.flatten(shape_feats, start_dim=1)

        pred_pose = self.pose_mlp(pose_feats)
        pred_cam = self.cam_mlp(shape_feats)
        pred_shape = self.shape_mlp(shape_feats)

        pred_pose = pred_pose.squeeze(-1).transpose(2, 1) # N, J, 6
        return pred_pose, pred_shape, pred_cam

    