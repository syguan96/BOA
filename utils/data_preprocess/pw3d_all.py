import os
import cv2
import numpy as np
import pickle
import ipdb
from tqdm import tqdm
import torch

from utils.geometry import batch_rodrigues
from utils.smpl import SMPL
import config

# --- predefined variables
# smpl = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False)
device = torch.device('cpu')
smpl_male = SMPL(config.SMPL_MODEL_DIR, gender='male', create_transl=False).to(device)
smpl_female = SMPL(config.SMPL_MODEL_DIR, gender='female', create_transl=False).to(device)
# --- predefined variables end

def projection(smpl, smpl_trans, camPose, camIntrinsics):
    smpl += smpl_trans
    smpl = np.concatenate([smpl, np.ones((49, 1))], axis=1)
    smpl = np.dot(smpl, camPose.T)[:, :3]
    smpl /= smpl[:, np.newaxis, -1]
    smpl = np.dot(smpl, camIntrinsics.T)
    return smpl[:,:2]

def get_smpl_joints(gt_betas, gt_pose, gender):
    gt_betas = torch.from_numpy(gt_betas).float().unsqueeze(0)
    gt_pose = torch.from_numpy(gt_pose).float().unsqueeze(0)
    gt_joints = smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).joints 
    gt_joints_female = smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).joints
    gt_joints[gender==1, :, :] = gt_joints_female[gender==1, :, :]
    gt_joints = gt_joints.squeeze().numpy()
    return gt_joints

def pw3d_extract(dataset_path, out_path):

    openpose_coco2common = [10, 9, 8, 11, 12, 13, 4, 3, 2, 5, 6, 7]
    common2J24 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # scale factor
    scaleFactor = 1.2

    # structs we use
    imgnames_, scales_, centers_, parts_ = [], [], [], []
    poses_, shapes_, genders_ = [], [], [] 
    smpl_j3ds_, smpl_j2ds_ = [], []

    # get a list of .pkl files in the directory
    dataset_path = os.path.join(dataset_path, 'sequenceFiles', 'test')
    tmp = ['downtown_runForBus_00.pkl', 'downtown_rampAndStairs_00.pkl', 'flat_packBags_00.pkl', 'downtown_runForBus_01.pkl', 'office_phoneCall_00.pkl', 'downtown_windowShopping_00.pkl', 'downtown_walkUphill_00.pkl', 'downtown_sitOnStairs_00.pkl', 'downtown_enterShop_00.pkl', 'downtown_walking_00.pkl', 'downtown_stairs_00.pkl', 'downtown_crossStreets_00.pkl', 'downtown_car_00.pkl', 'downtown_downstairs_00.pkl', 'downtown_bar_00.pkl', 'downtown_walkBridge_01.pkl', 'downtown_weeklyMarket_00.pkl', 'downtown_warmWelcome_00.pkl', 'downtown_arguing_00.pkl', 'downtown_upstairs_00.pkl', 'downtown_bus_00.pkl', 'flat_guitar_01.pkl', 'downtown_cafe_00.pkl', 'outdoors_fencing_01.pkl']
    files = [os.path.join(dataset_path, f) 
        for f in tmp]
    # go through all the .pkl files
    for filename in tqdm(files):
        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            smpl_pose = data['poses']
            smpl_betas = data['betas']
            poses2d = data['poses2d']
            global_poses = data['cam_poses']
            genders = data['genders']
            valid = np.array(data['campose_valid']).astype(np.bool)
            num_people = len(smpl_pose)
            num_frames = len(smpl_pose[0])
            seq_name = str(data['sequence'])
            img_names = np.array(['imageFiles/' + seq_name + '/image_%s.jpg' % str(i).zfill(5) for i in range(num_frames)])
            
            trans = data['trans']
            cam_Intrinsics = data['cam_intrinsics']

            # get through all the people in the sequence

            for i in range(num_people):
                valid_pose = smpl_pose[i][valid[i]]
                valid_betas = np.tile(smpl_betas[i][:10].reshape(1,-1), (num_frames, 1))
                valid_betas = valid_betas[valid[i]]
                valid_keypoints_2d = poses2d[i][valid[i]]
                valid_img_names = img_names[valid[i]]
                valid_global_poses = global_poses[valid[i]]
                gender = genders[i]

                valid_trans = trans[i][valid[i]]
                # consider only valid frames
                for valid_i in range(valid_pose.shape[0]):
                    part = valid_keypoints_2d[valid_i,:,:].T
                    part_ori = part.copy()
                    part = part[part[:,2]>0,:]
                    # bbox = [min(part[:,0]), min(part[:,1]),
                    #     max(part[:,0]), max(part[:,1])]
                    # center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                    # scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200
                    
                    # collect 3d joints (smpl format) and 2d joints (49 points)
                    smpl_j3d = get_smpl_joints(valid_betas[valid_i], valid_pose[valid_i], gender)
                    smpl_j2d = projection(smpl_j3d, valid_trans[valid_i], valid_global_poses[valid_i], cam_Intrinsics)
                    smpl_j2d = np.concatenate([smpl_j2d, np.ones((49,1))], axis=1)
                    bbox = [min(smpl_j2d[:,0]), min(smpl_j2d[:,1]),
                        max(smpl_j2d[:,0]), max(smpl_j2d[:,1])]
                    center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                    scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200

                    # convert kp2d
                    j2d = np.zeros([24,3])
                    part_ori[:, 2] = part_ori[:, 2] > 0.3
                    j12 = part_ori[openpose_coco2common]
                    j2d[common2J24,:] = j12[:,:]
                    # j2d[common2J24,2] = 1

                    # transform global pose
                    pose = valid_pose[valid_i]
                    extrinsics = valid_global_poses[valid_i][:3,:3]
                    pose[:3] = cv2.Rodrigues(np.dot(extrinsics, cv2.Rodrigues(pose[:3])[0]))[0].T[0]                      

                    imgnames_.append(valid_img_names[valid_i])
                    centers_.append(center)
                    scales_.append(scale)
                    poses_.append(pose)
                    shapes_.append(valid_betas[valid_i])
                    genders_.append(gender)
                    parts_.append(j2d)
                    smpl_j3ds_.append(smpl_j3d)
                    smpl_j2ds_.append(smpl_j2d)
                    # ipdb.set_trace()

    # store data
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path,
        '3dpw_test.npz')
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       pose=poses_,
                       shape=shapes_,
                       part=parts_,
                       gender=genders_,
                       smpl_j3d=smpl_j3ds_,
                       smpl_j2d=smpl_j2ds_,
                       )
