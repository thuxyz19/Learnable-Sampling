from __future__ import absolute_import, division, print_function
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import argparse
import os
import os.path as op
import time
import datetime
import torch
from torchvision.utils import make_grid
import gc
import numpy as np
import cv2
from src.modeling.model import get_model
from src.modeling._smpl import SMPL
import src.utils.config as cfg
from src.datasets.build import make_data_loader, make_data_loader_tsv
from src.utils.vibe_renderer import Renderer
from src.utils.logger import setup_logger
from src.utils.comm import synchronize, is_main_process, get_rank, all_gather
from src.utils.miscellaneous import mkdir, set_seed
from src.utils.metric_logger import AverageMeter, EvalMetricsLogger
from src.utils.renderer import visualize_reconstruction, visualize_reconstruction_heat
from src.utils.metric_pampjpe import reconstruction_error
from src.utils.geometric_layers import orthographic_projection, batch_rodrigues
import torch.nn.functional as F
from azureml.core.run import Run
aml_run = Run.get_context()

criterion_keypoints = torch.nn.MSELoss(reduction='none')
criterion_regr = torch.nn.MSELoss(reduction='none')

def smpl_losses(pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl=None, pose_conf=None):
    pred_rotmat_valid = pred_rotmat.reshape(-1, 24, 3, 3)
    gt_rotmat_valid = batch_rodrigues(gt_pose.reshape(-1,3)).reshape(-1, 24, 3, 3)
    pred_betas_valid = pred_betas
    gt_betas_valid = gt_betas
    pred_rotmat_valid = pred_rotmat_valid[has_smpl == 1]
    gt_rotmat_valid = gt_rotmat_valid[has_smpl == 1]
    pred_betas_valid = pred_betas_valid[has_smpl == 1]
    gt_betas_valid = gt_betas_valid[has_smpl == 1]
    if pose_conf is not None:
        pose_conf = pose_conf.unsqueeze(-1).unsqueeze(-1)
    else:
        pose_conf = torch.ones_like(gt_rotmat_valid, device=gt_rotmat_valid.device, dtype=gt_rotmat_valid.dtype)
    if len(pred_rotmat_valid) > 0:
        loss_regr_pose = (pose_conf * criterion_regr(pred_rotmat_valid, gt_rotmat_valid)).mean()
        loss_regr_betas = criterion_regr(pred_betas_valid, gt_betas_valid).mean()
    else:
        loss_regr_pose = torch.FloatTensor(1).fill_(0.).cuda()
        loss_regr_betas = torch.FloatTensor(1).fill_(0.).cuda()
    return loss_regr_pose, loss_regr_betas

def save_checkpoint(model, args, epoch, iteration, num_trial=10):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, iteration))
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            torch.save(model_to_save, op.join(checkpoint_dir, 'model.bin'))
            torch.save(model_to_save.state_dict(), op.join(checkpoint_dir, 'state_dict.bin'))
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir

def mean_per_joint_position_error(pred, gt, has_3d_joints):
    """ 
    Compute mPJPE
    """
    gt = gt[has_3d_joints == 1]
    gt = gt[:, :, :-1]
    pred = pred[has_3d_joints == 1]

    with torch.no_grad():
        gt_pelvis = (gt[:, 2,:] + gt[:, 3,:]) / 2
        gt = gt - gt_pelvis[:, None, :]
        pred_pelvis = (pred[:, 2,:] + pred[:, 3,:]) / 2
        pred = pred - pred_pelvis[:, None, :]
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error

def mean_per_vertex_error(pred, gt, has_smpl):
    """
    Compute mPVE
    """
    pred = pred[has_smpl == 1]
    gt = gt[has_smpl == 1]
    with torch.no_grad():
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error

def keypoint_2d_loss(criterion_keypoints, pred_keypoints_2d, gt_keypoints_2d, has_pose_2d):
    """
    Compute 2D reprojection loss if 2D keypoint annotations are available.
    The confidence (conf) is binary and indicates whether the keypoints exist or not.
    """
    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    loss = (conf * criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
    return loss

def keypoint_3d_loss(criterion_keypoints, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d, device):
    """
    Compute 3D keypoint loss if 3D keypoint annotations are available.
    """
    conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
    gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
    conf = conf[has_pose_3d == 1]
    pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
    if len(gt_keypoints_3d) > 0:
        gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
        pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
        return (conf * criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
    else:
        return torch.FloatTensor(1).fill_(0.).to(device) 

def vertices_loss(criterion_vertices, pred_vertices, gt_vertices, has_smpl, device):
    """
    Compute per-vertex loss if vertex annotations are available.
    """
    pred_vertices_with_shape = pred_vertices[has_smpl == 1]
    gt_vertices_with_shape = gt_vertices[has_smpl == 1]
    if len(gt_vertices_with_shape) > 0:
        return criterion_vertices(pred_vertices_with_shape, gt_vertices_with_shape)
    else:
        return torch.FloatTensor(1).fill_(0.).to(device) 
    
def visualize_heat(images,
                    gt_keypoints_2d,
                    pred_keypoints_2d,
                    heatmaps):
    """Tensorboard logging."""
    gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
    to_lsp = list(range(14))
    rend_imgs = []
    batch_size = gt_keypoints_2d.shape[0]
    if len(heatmaps.shape) == 4:
        # heatmaps = F.softmax(heatmaps.view(batch_size, 24, -1), dim=-1)
        heatmaps = heatmaps.view(batch_size, 24, 56, 56)
        heatmaps = F.interpolate(heatmaps, (224, 224), mode='bilinear')
    else:
        # heatmaps = F.softmax(heatmaps.view(batch_size, 24, 9*9, -1), dim=-1)
        heatmaps = heatmaps.max(2)[0]
        heatmaps = heatmaps.view(batch_size, 24, 56, 56)
        heatmaps = F.interpolate(heatmaps, (224, 224), mode='bilinear')

    # heatmaps = heatmaps * 255.
    # Do visualization for the first 6 images of the batch
    for i in range(min(batch_size, 10)):
        img = images[i].cpu().numpy().transpose(1,2,0)
        heat = heatmaps[i].cpu().numpy().transpose(1,2,0)
        # Get LSP keypoints from the full list of keypoints
        gt_keypoints_2d_ = gt_keypoints_2d[i, to_lsp]
        pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[i, to_lsp]
        # Get predict vertices for the particular example
        # Visualize reconstruction and detected pose
        rend_img = visualize_reconstruction_heat(img, 224, gt_keypoints_2d_, pred_keypoints_2d_, heat)
        rend_img = rend_img.transpose(2,0,1)
        rend_imgs.append(torch.from_numpy(rend_img))   
    rend_imgs = make_grid(rend_imgs, nrow=1)
    return rend_imgs

def run(args, train_dataloader, val_dataloader, Graphormer_model, smpl, renderer, smpl_male=None):
    smpl.eval()
    max_iter = len(train_dataloader)
    iters_per_epoch = max_iter // args.num_train_epochs // 1

    optimizer = torch.optim.Adam(params=list(Graphormer_model.parameters()),
                                           lr=args.lr,
                                           betas=(0.9, 0.999),
                                           weight_decay=0)

    # define loss function (criterion) and optimizer
    criterion_2d_keypoints = torch.nn.MSELoss(reduction='none').cuda(args.device)
    criterion_keypoints = torch.nn.MSELoss(reduction='none').cuda(args.device)
    criterion_vertices = torch.nn.L1Loss().cuda(args.device)

    if args.distributed:
        Graphormer_model = torch.nn.parallel.DistributedDataParallel(
            Graphormer_model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

        logger.info(
                ' '.join(
                ['Local rank: {o}', 'Max iteration: {a}', 'iters_per_epoch: {b}','num_train_epochs: {c}',]
                ).format(o=args.local_rank, a=max_iter, b=iters_per_epoch, c=args.num_train_epochs)
            )

    start_training_time = time.time()
    end = time.time()
    Graphormer_model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    log_losses = AverageMeter()
    log_loss_2djoints = AverageMeter()
    log_loss_3djoints = AverageMeter()
    log_loss_vertices = AverageMeter()
    log_loss_pose = AverageMeter()
    log_loss_shape = AverageMeter()
    log_eval_metrics = EvalMetricsLogger()

    for iteration, (_, images, annotations) in enumerate(train_dataloader):
        if len(annotations['joints_3d'].shape) > 3:
            annotations['joints_3d'] = annotations['joints_3d'].view(annotations['joints_3d'].shape[0]*annotations['joints_3d'].shape[1], *annotations['joints_3d'].shape[2:])
            annotations['joints_2d'] = annotations['joints_2d'].view(annotations['joints_2d'].shape[0]*annotations['joints_2d'].shape[1], *annotations['joints_2d'].shape[2:])
            annotations['has_3d_joints'] = annotations['has_3d_joints'].view(annotations['has_3d_joints'].shape[0]*annotations['has_3d_joints'].shape[1], *annotations['has_3d_joints'].shape[2:])
            annotations['has_3d_joints'] = annotations['has_3d_joints'].squeeze(-1)
            annotations['has_2d_joints'] = annotations['has_2d_joints'].view(annotations['has_2d_joints'].shape[0]*annotations['has_2d_joints'].shape[1], *annotations['has_2d_joints'].shape[2:])
            annotations['has_2d_joints'] = annotations['has_2d_joints'].squeeze(-1)
            annotations['pose'] = annotations['pose'].view(annotations['pose'].shape[0]*annotations['pose'].shape[1], *annotations['pose'].shape[2:])
            annotations['betas'] = annotations['betas'].view(annotations['betas'].shape[0]*annotations['betas'].shape[1], *annotations['betas'].shape[2:])
            annotations['has_smpl'] = annotations['has_smpl'].view(annotations['has_smpl'].shape[0]*annotations['has_smpl'].shape[1], *annotations['has_smpl'].shape[2:])
            annotations['has_smpl'] = annotations['has_smpl'].squeeze(-1)
            annotations['ori_img'] = annotations['ori_img'].view(annotations['ori_img'].shape[0]*annotations['ori_img'].shape[1], *annotations['ori_img'].shape[2:])
        Graphormer_model.train()
        iteration += 1
        epoch = iteration // iters_per_epoch
        batch_size = images.size(0)
        data_time.update(time.time() - end)
        images = images.cuda(args.device)
        if len(images.shape) < 5:
            images = images.unsqueeze(1) 
        gt_2d_joints = annotations['joints_2d'].cuda(args.device)
        if gt_2d_joints.shape[1] > 14: 
            gt_2d_joints = gt_2d_joints[:,cfg.J24_TO_J14,:]
        has_2d_joints = annotations['has_2d_joints'].cuda(args.device)
        gt_3d_joints = annotations['joints_3d'].cuda(args.device)
        if gt_3d_joints.shape[1] > 14: 
            gt_3d_pelvis = gt_3d_joints[:,cfg.J24_NAME.index('Pelvis'),:3]
            gt_3d_joints = gt_3d_joints[:,cfg.J24_TO_J14,:] 
            gt_3d_joints[:,:,:3] = gt_3d_joints[:,:,:3] - gt_3d_pelvis[:, None, :]
        has_3d_joints = annotations['has_3d_joints'].cuda(args.device)
        gt_pose = annotations['pose'].cuda(args.device)
        gt_betas = annotations['betas'].cuda(args.device)
        has_smpl = annotations['has_smpl'].cuda(args.device)
        gt_vertices = smpl(gt_pose, gt_betas)
        # normalize gt based on smpl's pelvis 
        gt_smpl_3d_joints = smpl.get_h36m_joints(gt_vertices)
        gt_smpl_3d_pelvis = gt_smpl_3d_joints[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
        gt_vertices = gt_vertices - gt_smpl_3d_pelvis[:, None, :]
        # forward-pass
        pred_camera, pred_rotmat, pred_shape, pred_heat = Graphormer_model(images, return_heat=True)
        pred_shape = pred_shape.view(-1, args.seq, 10).mean(1, keepdim=True).repeat(1, args.seq, 1).view(-1, 10)
        # obtain 3d joints, which are regressed from the full mesh
        pred_vertices = smpl(pred_rotmat, pred_shape)
        pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)
        pred_3d_joints_smpl_pelvis = pred_3d_joints_from_smpl[:, cfg.H36M_J17_NAME.index('Pelvis'), :]
        pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,cfg.H36M_J17_TO_J14,:]
        pred_3d_joints_from_smpl_loss = pred_3d_joints_from_smpl - pred_3d_joints_smpl_pelvis[:, None, :]
        # obtain 2d joints, which are projected from 3d joints of smpl mesh
        pred_2d_joints_from_smpl = orthographic_projection(pred_3d_joints_from_smpl, pred_camera)
        pred_vertices_loss = pred_vertices - pred_3d_joints_smpl_pelvis[:, None, :]
        # compute 3d vertex loss
        loss_vertices = vertices_loss(criterion_vertices, pred_vertices_loss, gt_vertices, has_smpl, args.device)
        # compute 3d joint loss (where the joints are regressed from full mesh)
        loss_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints_from_smpl_loss, gt_3d_joints, has_3d_joints, args.device)
        # compute 2d joint loss
        loss_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_smpl, gt_2d_joints, has_2d_joints)
        # compute smpl param loss
        loss_smpl_pose, loss_smpl_shape = smpl_losses(pred_rotmat, pred_shape, gt_pose, gt_betas, has_smpl)
        # we empirically use hyperparameters to balance difference losses
        loss = args.joints_loss_weight*loss_3d_joints + \
                args.vertices_loss_weight*loss_vertices  + args.joints_loss_weight_2d * loss_2d_joints + args.pose_loss_weight* loss_smpl_pose + \
                    args.shape_loss_weight*loss_smpl_shape
        # update logs
        log_loss_2djoints.update(loss_2d_joints.item(), batch_size)
        log_loss_3djoints.update(loss_3d_joints.item(), batch_size)
        log_loss_vertices.update(loss_vertices.item(), batch_size)
        log_loss_pose.update(loss_smpl_pose.item(), batch_size)
        log_loss_shape.update(loss_smpl_shape.item(), batch_size)
        log_losses.update(loss.item(), batch_size)

        # back prop
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if iteration % args.logging_steps == 0 or iteration == max_iter:
            eta_seconds = batch_time.avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                ' '.join(
                ['eta: {eta}', 'epoch: {ep}', 'iter: {iter}', 'max mem : {memory:.0f}',]
                ).format(eta=eta_string, ep=epoch, iter=iteration, 
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0) 
                + '  loss: {:.4f}, 2d joint loss: {:.4f}, 3d joint loss: {:.4f}, vertex loss: {:.4f}, smpl_pose loss: {:.4f}, smpl_shape loss: {:.4f}, compute: {:.4f}, data: {:.4f}, lr: {:.6f}'.format(
                    log_losses.avg, log_loss_2djoints.avg, log_loss_3djoints.avg, log_loss_vertices.avg, log_loss_pose.avg, log_loss_shape.avg, batch_time.avg, data_time.avg, 
                    optimizer.param_groups[0]['lr'])
            )

            aml_run.log(name='Loss', value=float(log_losses.avg))
            aml_run.log(name='3d joint Loss', value=float(log_loss_3djoints.avg))
            aml_run.log(name='2d joint Loss', value=float(log_loss_2djoints.avg))
            aml_run.log(name='vertex Loss', value=float(log_loss_vertices.avg))
            aml_run.log(name='smpl pose Loss', value=float(log_loss_pose.avg))
            aml_run.log(name='smpl shape Loss', value=float(log_loss_shape.avg))

            visual_imgs = visualize_mesh(renderer,
                                            annotations['ori_img'].detach(),
                                            annotations['joints_2d'].detach(),
                                            pred_vertices.detach(), 
                                            pred_camera.detach(),
                                            pred_2d_joints_from_smpl.detach())
            visual_imgs = visual_imgs.transpose(0,1)
            visual_imgs = visual_imgs.transpose(1,2)
            visual_imgs = np.asarray(visual_imgs)

            if is_main_process()==True:
                stamp = str(epoch) + '_' + str(iteration)
                temp_fname = args.output_dir + 'visual_' + stamp + '.jpg'
                cv2.imwrite(temp_fname, np.asarray(visual_imgs[:,:,::-1]*255))
                aml_run.log_image(name='visual results', path=temp_fname)
            
            visual_imgs = visualize_heat(annotations['ori_img'].detach(),
                                        annotations['joints_2d'].detach(),
                                        pred_2d_joints_from_smpl.detach(),
                                        pred_heat.detach())
            visual_imgs = visual_imgs.transpose(0,1)
            visual_imgs = visual_imgs.transpose(1,2)
            visual_imgs = np.asarray(visual_imgs)
            if is_main_process()==True:
                temp_fname = args.output_dir + 'visual_debug_' + stamp + '.jpg'
                cv2.imwrite(temp_fname, np.asarray(visual_imgs[:,:,::-1]*255))

        if iteration % iters_per_epoch == 0:
            val_mPVE, val_mPJPE, val_PAmPJPE, val_count = run_validate(args, val_dataloader, 
                                                Graphormer_model, 
                                                smpl,
                                                smpl_male=smpl_male)

            aml_run.log(name='mPVE', value=float(1000*val_mPVE))
            aml_run.log(name='mPJPE', value=float(1000*val_mPJPE))
            aml_run.log(name='PAmPJPE', value=float(1000*val_PAmPJPE))
            logger.info(
                ' '.join(['Validation', 'epoch: {ep}',]).format(ep=epoch) 
                + '  mPVE: {:6.2f}, mPJPE: {:6.2f}, PAmPJPE: {:6.2f}, Data Count: {:6.2f}'.format(1000*val_mPVE, 1000*val_mPJPE, 1000*val_PAmPJPE, val_count)
            )
            if val_PAmPJPE<log_eval_metrics.PAmPJPE or epoch % 10 == 0:
                checkpoint_dir = save_checkpoint(Graphormer_model, args, epoch, iteration)
            if val_PAmPJPE<log_eval_metrics.PAmPJPE:
                log_eval_metrics.update(val_mPVE, val_mPJPE, val_PAmPJPE, epoch)
                
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info('Total training time: {} ({:.4f} s / iter)'.format(
        total_time_str, total_training_time / max_iter)
    )
    checkpoint_dir = save_checkpoint(Graphormer_model, args, epoch, iteration)

    logger.info(
        ' Best Results:'
        + '  mPVE: {:6.2f}, mPJPE: {:6.2f}, PAmPJPE: {:6.2f}, at epoch {:6.2f}'.format(1000*log_eval_metrics.mPVE, 1000*log_eval_metrics.mPJPE, 1000*log_eval_metrics.PAmPJPE, log_eval_metrics.epoch)
    )

def run_eval_general(args, val_dataloader, Graphormer_model, smpl, smpl_male=None):
    smpl.eval()
    epoch = 0
    if args.distributed:
        Graphormer_model = torch.nn.parallel.DistributedDataParallel(
            Graphormer_model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    Graphormer_model.eval()

    val_mPVE, val_mPJPE, val_PAmPJPE, _ = run_validate(args, val_dataloader, 
                                    Graphormer_model, 
                                    smpl,
                                    smpl_male=smpl_male)

    aml_run.log(name='mPVE', value=float(1000*val_mPVE))
    aml_run.log(name='mPJPE', value=float(1000*val_mPJPE))
    aml_run.log(name='PAmPJPE', value=float(1000*val_PAmPJPE))


    logger.info(
        ' '.join(['Validation', 'epoch: {ep}',]).format(ep=epoch) 
        + '  mPVE: {:6.2f}, mPJPE: {:6.2f}, PAmPJPE: {:6.2f}'.format(1000*val_mPVE, 1000*val_mPJPE, 1000*val_PAmPJPE)
    )

def run_validate(args, val_loader, Graphormer_model, smpl, smpl_male=None):
    mPVE = AverageMeter()
    mPJPE = AverageMeter()
    PAmPJPE = AverageMeter()
    # switch to evaluate mode
    Graphormer_model.eval()
    smpl.eval()
    if smpl_male is not None:
        smpl_male.eval()
    with torch.no_grad():
        # end = time.time()
        for i, (_, images, annotations) in enumerate(val_loader):
            if len(annotations['joints_3d'].shape) > 3:
                annotations['joints_3d'] = annotations['joints_3d'].view(annotations['joints_3d'].shape[0]*annotations['joints_3d'].shape[1], *annotations['joints_3d'].shape[2:])
                annotations['joints_2d'] = annotations['joints_2d'].view(annotations['joints_2d'].shape[0]*annotations['joints_2d'].shape[1], *annotations['joints_2d'].shape[2:])
                annotations['has_3d_joints'] = annotations['has_3d_joints'].view(annotations['has_3d_joints'].shape[0]*annotations['has_3d_joints'].shape[1], *annotations['has_3d_joints'].shape[2:])
                annotations['has_3d_joints'] = annotations['has_3d_joints'].squeeze(-1)
                annotations['has_2d_joints'] = annotations['has_2d_joints'].view(annotations['has_2d_joints'].shape[0]*annotations['has_2d_joints'].shape[1], *annotations['has_2d_joints'].shape[2:])
                annotations['has_2d_joints'] = annotations['has_2d_joints'].squeeze(-1)
                annotations['pose'] = annotations['pose'].view(annotations['pose'].shape[0]*annotations['pose'].shape[1], *annotations['pose'].shape[2:])
                annotations['betas'] = annotations['betas'].view(annotations['betas'].shape[0]*annotations['betas'].shape[1], *annotations['betas'].shape[2:])
                annotations['has_smpl'] = annotations['has_smpl'].view(annotations['has_smpl'].shape[0]*annotations['has_smpl'].shape[1], *annotations['has_smpl'].shape[2:])
                annotations['has_smpl'] = annotations['has_smpl'].squeeze(-1)
                annotations['ori_img'] = annotations['ori_img'].view(annotations['ori_img'].shape[0]*annotations['ori_img'].shape[1], *annotations['ori_img'].shape[2:])
            # compute output
            images = images.cuda(args.device)
            if len(images.shape) < 5:
                images = images.unsqueeze(1)
            gt_3d_joints = annotations['joints_3d'].cuda(args.device)
            if gt_3d_joints.shape[1] > 14:
                gt_3d_pelvis = gt_3d_joints[:,cfg.J24_NAME.index('Pelvis'),:3]
                gt_3d_joints = gt_3d_joints[:,cfg.J24_TO_J14,:] 
                gt_3d_joints[:,:,:3] = gt_3d_joints[:,:,:3] - gt_3d_pelvis[:, None, :]
            gt_2d_joints = annotations['joints_2d'].cuda(args.device)
            if gt_2d_joints.shape[1] > 14: 
                gt_2d_joints = gt_2d_joints[:,cfg.J24_TO_J14,:]
            has_3d_joints = annotations['has_3d_joints'].cuda(args.device)
            gt_pose = annotations['pose'].cuda(args.device)
            gt_betas = annotations['betas'].cuda(args.device)
            has_smpl = annotations['has_smpl'].cuda(args.device)
            if smpl_male is None:
                gt_vertices = smpl(gt_pose, gt_betas)
                gt_smpl_3d_joints = smpl.get_h36m_joints(gt_vertices)
            else:
                gt_vertices = smpl_male(gt_pose, gt_betas)
                gt_smpl_3d_joints = smpl_male.get_h36m_joints(gt_vertices)
            gt_smpl_3d_pelvis = gt_smpl_3d_joints[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
            gt_vertices = gt_vertices - gt_smpl_3d_pelvis[:, None, :] 
            gt_smpl_3d_joints = gt_smpl_3d_joints[:,cfg.H36M_J17_TO_J14,:] 
            gt_smpl_3d_joints[:, :, :3] = gt_smpl_3d_joints[:, :, :3] - gt_smpl_3d_pelvis[:, None, :]
            gt_smpl_3d_joints = torch.cat([gt_smpl_3d_joints, gt_3d_joints[:, :, -1:]], -1)
            # forward-pass
            _, pred_rotmat, pred_shape, _ = Graphormer_model(images)
            pred_shape = pred_shape.view(-1, args.seq, 10).mean(1, keepdim=True).repeat(1, args.seq, 1).view(-1, 10)
            # obtain 3d/2d joints
            pred_vertices = smpl(pred_rotmat, pred_shape)
            pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)
            pred_3d_pelvis = pred_3d_joints_from_smpl[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
            pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,cfg.H36M_J17_TO_J14,:]
            pred_3d_joints_from_smpl = pred_3d_joints_from_smpl - pred_3d_pelvis[:, None, :]
            pred_vertices = pred_vertices - pred_3d_pelvis[:, None, :]
            # measure errors
            error_vertices = mean_per_vertex_error(pred_vertices, gt_vertices, has_smpl)
            error_joints = mean_per_joint_position_error(pred_3d_joints_from_smpl, gt_3d_joints,  has_3d_joints)
            error_joints_pa, _ = reconstruction_error(pred_3d_joints_from_smpl.cpu().numpy(), gt_3d_joints[:,:,:3].cpu().numpy(), reduction=None, return_scale=True)
            if len(error_vertices)>0:
                mPVE.update(np.mean(error_vertices), int(torch.sum(has_smpl)) )
            if len(error_joints)>0:
                mPJPE.update(np.mean(error_joints), int(has_3d_joints.shape[0]) )  
            if len(error_joints_pa)>0:
                PAmPJPE.update(np.mean(error_joints_pa), int(has_3d_joints.shape[0]) )

    val_mPVE = all_gather(float(mPVE.avg))
    val_mPVE = sum(val_mPVE)/len(val_mPVE)
    val_mPJPE = all_gather(float(mPJPE.avg))
    val_mPJPE = sum(val_mPJPE)/len(val_mPJPE)
    val_PAmPJPE = all_gather(float(PAmPJPE.avg))
    val_PAmPJPE = sum(val_PAmPJPE)/len(val_PAmPJPE)
    val_count = all_gather(float(mPVE.count))
    val_count = sum(val_count)

    return val_mPVE, val_mPJPE, val_PAmPJPE, val_count

def visualize_mesh(renderer,
                    images,
                    gt_keypoints_2d,
                    pred_vertices, 
                    pred_camera,
                    pred_keypoints_2d):
    """Tensorboard logging."""
    gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
    to_lsp = list(range(14))
    rend_imgs = []
    batch_size = pred_vertices.shape[0]
    # Do visualization for the first 10 images of the batch
    for i in range(min(batch_size, 10)):
        img = images[i].cpu().numpy().transpose(1,2,0)
        # Get LSP keypoints from the full list of keypoints
        gt_keypoints_2d_ = gt_keypoints_2d[i, to_lsp]
        pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[i, to_lsp]
        # Get predict vertices for the particular example
        vertices = pred_vertices[i].cpu().numpy()
        cam = pred_camera[i].cpu().numpy()
        # Visualize reconstruction and detected pose
        rend_img = visualize_reconstruction(img, 224, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam, renderer, att_2d=None)
        rend_img = rend_img.transpose(2,0,1)
        rend_imgs.append(torch.from_numpy(rend_img))   
    rend_imgs = make_grid(rend_imgs, nrow=1)
    return rend_imgs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", default='experiments/3dpw_config.yaml', type=str, required=False,
                        help="Yaml file for the model configuration")
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--data_dir", default='datasets', type=str, required=False,
                        help="Directory with all datasets, each in one subfolder")
    parser.add_argument("--train_yaml", default='imagenet2012/train.yaml', type=str, required=False,
                        help="Yaml file with all data for training.")
    parser.add_argument("--val_yaml", default='imagenet2012/test.yaml', type=str, required=False,
                        help="Yaml file with all data for validation.")      
    parser.add_argument("--num_workers", default=4, type=int, 
                        help="Workers in dataloader.")
    parser.add_argument("--img_scale_factor", default=1, type=int, 
                        help="adjust image resolution.") 
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--resume_checkpoint", default=None, type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    #########################################################
    # Training parameters
    #########################################################
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int, 
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--lr', "--learning_rate", default=5e-5, type=float, 
                        help="The initial lr.")
    parser.add_argument("--num_train_epochs", default=200, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--vertices_loss_weight", default=0.0, type=float)          
    parser.add_argument("--joints_loss_weight", default=300.0, type=float)
    parser.add_argument("--joints_loss_weight_2d", default=300.0, type=float)
    parser.add_argument("--pose_loss_weight", default=60.0, type=float)
    parser.add_argument("--shape_loss_weight", default=0.06, type=float)
    parser.add_argument("--drop_out", default=0.1, type=float, 
                        help="Drop out ratio in BERT.")
    #########################################################
    # Others
    #########################################################
    parser.add_argument("--run_eval_only", default=False, action='store_true',) 
    parser.add_argument('--logging_steps', type=int, default=100, 
                        help="Log every X steps.")
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88, 
                        help="random seed for initialization.")
    parser.add_argument('--seq', type=int, default=1, 
                        help="the seq length of a sample.")
    parser.add_argument("--local_rank", type=int, default=0, 
                        help="For distributed training.")

    args = parser.parse_args()
    return args

def main(args):
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    os.environ['OMP_NUM_THREADS'] = str(args.num_workers)
    print('set os.environ[OMP_NUM_THREADS] to {}'.format(os.environ['OMP_NUM_THREADS']))
    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)
    if args.distributed:
        print("Init distributed training on local rank {} ({}), rank {}, world size {}".format(args.local_rank, int(os.environ["LOCAL_RANK"]), int(os.environ["LOCAL_RANK"]), args.num_gpus))
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://'
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        args.device = torch.device("cuda", local_rank)
        synchronize()
    mkdir(args.output_dir)
    logger = setup_logger("Graphormer", args.output_dir, get_rank())
    set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))
    # Renderer for visualization
    renderer = Renderer(
            resolution=(224, 224),
            orig_img=True,
            wireframe=False
        )

    # Load model
    from src.utils.config import update_hparams
    config = update_hparams(args.config_yaml)
    _model = get_model(config)
    smpl = SMPL().to(args.device)
    smpl_male = SMPL(gender='m').to(args.device)
    if args.resume_checkpoint!=None and args.resume_checkpoint!='None':
        # for fine-tuning or resume training or inference, load weights from checkpoint
        logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
        # workaround approach to load sparse tensor in graph conv.
        states = torch.load(args.resume_checkpoint, map_location='cpu')
        # states = checkpoint_loaded.state_dict()
        for k, v in states.items():
            states[k] = v.cpu()
        # del checkpoint_loaded
        _model.load_state_dict(states, strict=False)
        del states
        gc.collect()
        torch.cuda.empty_cache()
    _model.to(args.device)

    logger.info("Training parameters %s", args)
    if args.run_eval_only==True:
        if config.DATASET == '3dpw':
            val_dataloader = make_data_loader(args, args.val_yaml, 
                                            args.distributed, is_train=False, scale_factor=args.img_scale_factor)
        elif config.DATASET == 'h36m':
            val_dataloader = make_data_loader_tsv(args, args.val_yaml, 
                                            args.distributed, is_train=False, scale_factor=args.img_scale_factor)
        else:
            raise Exception('Only support 3DPW dataset and H36M dataset!')
        run_eval_general(args, val_dataloader, _model, smpl, smpl_male=smpl_male)

    else:
        if config.DATASET == '3dpw':
            train_dataloader = make_data_loader(args, args.train_yaml, 
                                                args.distributed, is_train=True, scale_factor=args.img_scale_factor)
            val_dataloader = make_data_loader(args, args.val_yaml, 
                                            args.distributed, is_train=False, scale_factor=args.img_scale_factor)
        elif config.DATASET == 'h36m':
            train_dataloader = make_data_loader_tsv(args, args.train_yaml, 
                                                args.distributed, is_train=True, scale_factor=args.img_scale_factor)
            val_dataloader = make_data_loader_tsv(args, args.val_yaml, 
                                            args.distributed, is_train=False, scale_factor=args.img_scale_factor)
        else:
            raise Exception('Only support 3DPW dataset and H36M dataset!')
        run(args, train_dataloader, val_dataloader, _model, smpl, renderer)

if __name__ == "__main__":
    args = parse_args()
    main(args)
