
import numpy as np
import torch
import torch.nn.functional as F

import config
import constants
from utils.geometry import perspective_projection, rotation_matrix_to_angle_axis
from utils.smplify.prior import MaxMixturePrior
from utils.smpl import SMPL

# predefined variables
device = torch.device('cuda')
J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
smpl_neutral = SMPL(config.SMPL_MODEL_DIR, create_transl=False).to(device)
smpl_male = SMPL(config.SMPL_MODEL_DIR, gender='male', create_transl=False).to(device)
smpl_female = SMPL(config.SMPL_MODEL_DIR, gender='female', create_transl=False).to(device)
poseprior = MaxMixturePrior(prior_folder='data/spin_data', num_gaussians=8, dtype=torch.float32).to(device)
# -- end


def decode_smpl_params(rotmats, betas, cam, neutral=True, pose2rot=False):
    if neutral:
        smpl_out = smpl_neutral(betas=betas, body_pose=rotmats[:,1:], global_orient=rotmats[:,0].unsqueeze(1), pose2rot=pose2rot)
    return {'s3d': smpl_out.joints, 'vts': smpl_out.vertices}

def projection(cam, s3d, eps=1e-9):
    cam_t = torch.stack([cam[:,1], cam[:,2],
                        2*constants.FOCAL_LENGTH/(constants.IMG_RES * cam[:,0] + eps)],dim=-1)
    camera_center = torch.zeros(s3d.shape[0], 2, device=device)
    s2d = perspective_projection(s3d,
                                rotation=torch.eye(3, device=device).unsqueeze(0).expand(s3d.shape[0], -1, -1),
                                translation=cam_t,
                                focal_length=constants.FOCAL_LENGTH,
                                camera_center=camera_center)
    s2d_norm = s2d / (constants.IMG_RES / 2.)  # to [-1,1]
    return {'ori':s2d, 'normed': s2d_norm}

def cal_s2d_loss(pred_s3d, trg_s2d, pred_cam):
    """
        pred_s2d: (B, 49, 2)
        gt_s2d: (B, 49, 3)
        only calculate the later 24 joints, i.e., 25:
    """
    pred_s2d = projection(pred_cam, pred_s3d)['normed']
    conf = trg_s2d[:, 25:, -1].unsqueeze(-1).clone()
    loss = (conf * F.mse_loss(pred_s2d[:, 25:], trg_s2d[:, 25:, :-1], reduction='none')).mean()
    return loss

def cal_s3d_loss(pred_s3d, gt_s3d):
    """
    pred_s3d: (B, 49, 3)
    gt_s3d: (B, 49, 4)

    """
    conf = gt_s3d[:,:,-1].unsqueeze(-1).clone()
    # gt_s3d = gt_s3d[:,25:]
    pred_s3d = pred_s3d[:,25:]
    # align the root
    gt_hip = (gt_s3d[:,2] + gt_s3d[:,3]) / 2
    gt_s3d = gt_s3d - gt_hip[:,None,:]
    pred_hip = (pred_s3d[:,2] + pred_s3d[:,3]) / 2
    pred_s3d = pred_s3d - pred_hip[:,None,:]
    # print(pred_s3d.shape, gt_s3d.shape, conf.shape)
    loss = (conf * F.mse_loss(pred_s3d, gt_s3d[:,:,:-1], reduction='none')).mean()
    return loss

def shape_prior(betas):
    shape_prior_loss = (betas ** 2).sum(dim=-1).mean()
    return shape_prior_loss

def pose_prior(pose, betas, angle_prior=False, gmm_prior=False):
    loss_items = {}
    body_pose = rotation_matrix_to_angle_axis(pose[:,1:].contiguous().view(-1,3,3)).contiguous().view(-1, 69)
    assert body_pose.shape[0] == pose.shape[0]
    if gmm_prior:
        pose_prior_loss = poseprior(body_pose, betas).mean()
        loss_items['gmm'] = pose_prior_loss
    if angle_prior:
        constant = torch.tensor([1., -1., -1, -1.]).to(device)
        angle_prior_loss = torch.exp(body_pose[:, [55-3, 58-3, 12-3, 15-3]] * constant) ** 2
        loss_items['angle'] = angle_prior_loss
    return loss_items

# def cal_motion_loss(pred_kps_t, pred_kps_n, gt_kps_t, gt_kps_n):
def cal_motion_loss(pred_s3d, pred_cam, pres_s3d_his, pred_cam_his, gt_kps_t, gt_kps_n):
    """
        pred_kps_t: (B, 49, 2), at time t
        pred_kps_n: (B, 49, 2), at time t-n
        gt_kps_t  : (B, 49, 3), at time t
        gt_kps_n  : (B, 49, 3), at time t-n
    """
    pred_kps_t = projection(pred_cam, pred_s3d)['normed']
    pred_kps_n = projection(pred_cam_his, pres_s3d_his)['normed']
    motion_pred = pred_kps_t[:, 25:] - pred_kps_n[:, 25:]
    motion_gt = gt_kps_t[:, 25:, :-1] - gt_kps_n[:, 25:, :-1]
    motion_loss = F.mse_loss(motion_pred, motion_gt)
    return motion_loss

def cal_consistent_constrain(pred_rotmat, pred_betas, pred_cam, ema_rotmat, ema_betas, ema_cam):
    smpl_out = decode_smpl_params(pred_rotmat, pred_betas, pred_cam, neutral=True, pose2rot=False)
    pred_s3d = smpl_out['s3d']
    pred_vts = smpl_out['vts']

    ema_smpl_out = decode_smpl_params(ema_rotmat, ema_betas, ema_cam, neutral=True, pose2rot=False)
    ema_pred_s3d = ema_smpl_out['s3d']
    ema_pred_vts = ema_smpl_out['vts']

    # kp loss
    proj_s2d_res = projection(pred_cam, pred_s3d)
    proj_s2d_normed = proj_s2d_res['normed']
    ema_proj_s2d_res = projection(ema_cam, ema_pred_s3d)
    ema_proj_s2d_normed = ema_proj_s2d_res['normed']
    s2ds_loss = cal_s2ds_loss_for_mt(proj_s2d_normed, ema_proj_s2d_normed)
    s3d_loss = cal_s3ds_loss_for_mt(pred_s3d, ema_pred_s3d)

    # smpl loss
    loss_pose = F.mse_loss(pred_rotmat, ema_rotmat)
    loss_beta = F.mse_loss(pred_betas, ema_betas)
    # loss = s3d_loss * self.options.consistent_s3d_weight + s2ds_loss * self.options.consistent_s2d_weight + \
    #         loss_pose * self.options.consistent_pose_weight + loss_beta * self.options.consistent_beta_weight
    return {'s3dloss': s3d_loss, 's2dloss': s2ds_loss, 'poseloss': loss_pose, 'betaloss':loss_beta}

def cal_s3ds_loss_for_mt(pred_s3d, gt_s3d):
    """
    pred_s3d: (B, 49, 3)
    gt_s3d: (B, 49, 4)
    """
    # conf = gt_s3d[:,:,-1].unsqueeze(-1).clone()
    gt_s3d = gt_s3d[:,25:]
    pred_s3d = pred_s3d[:,25:]
    # align the root
    gt_hip = (gt_s3d[:,2] + gt_s3d[:,3]) / 2
    gt_s3d = gt_s3d - gt_hip[:,None,:]
    pred_hip = (pred_s3d[:,2] + pred_s3d[:,3]) / 2
    pred_s3d = pred_s3d - pred_hip[:,None,:]
    loss = F.mse_loss(pred_s3d, gt_s3d)
    return loss

def cal_s2ds_loss_for_mt(pred_s2d, gt_s2d):
    """
    pred_s2d: (B, 49, 2)
    gt_s2d: (B, 49, 3)
    only calculate the later 24 joints, i.e., 25:
    """
    loss = F.mse_loss(pred_s2d[:,25:], gt_s2d[:,25:])
    return loss