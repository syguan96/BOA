
"""
This script is to processing PW3D. Note the processing procedure follows SPIN.
We split the test set according to action.
"""
import sys
sys.path.append('..')

import os
import cv2
import ipdb
import torch
import config
import pickle
import numpy as np
from tqdm import tqdm
from utils.smpl import SMPL
from utils.geometry import batch_rodrigues

# --- predefined variables
device = torch.device('cpu')
smpl_male = SMPL(config.SMPL_MODEL_DIR, gender='male', create_transl=False).to(device)
smpl_female = SMPL(config.SMPL_MODEL_DIR, gender='female', create_transl=False).to(device)

def projection(smpl, smpl_trans, camPose, camIntrinsics):
    """
    projection annoted 3D joints to 2D, so that we can obtain GT 2D joints.
    """
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

def pw3d_extract(dataset_path, out_path, debug=False):

    openpose_coco2common = [10, 9, 8, 11, 12, 13, 4, 3, 2, 5, 6, 7]
    common2J24 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # scale factor
    scaleFactor = 1.2

    shape_records = {}
    person_id = 0
    oldperson = False

    # get a list of .pkl files in the directory
    dataset_path = os.path.join(dataset_path, 'sequenceFiles', 'test')
    files = [os.path.join(dataset_path, f) 
        for f in os.listdir(dataset_path) if f.endswith('.pkl')]
    # go through all the .pkl files
    for fi, filename in enumerate(tqdm(files, desc='seq')):
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
            # structs we use
            imgnames_, scales_, centers_, parts_ = [], [], [], []
            poses_, shapes_, genders_ = [], [], [] 
            smpl_j3ds_, smpl_j2ds_ = [], []
            for i in range(num_people):
                valid_pose = smpl_pose[i][valid[i]]
                valid_betas = np.tile(smpl_betas[i][:10].reshape(1,-1), (num_frames, 1))
                valid_betas = valid_betas[valid[i]]
                valid_keypoints_2d = poses2d[i][valid[i]]
                valid_img_names = img_names[valid[i]]
                valid_global_poses = global_poses[valid[i]]
                gender = genders[i]

                # get the person id
                if fi ==0 and i==0 and person_id == 0:
                    shape_records[person_id] = smpl_betas[i][:10]
                    current_person = person_id
                else:
                    for hased_personid in shape_records.keys():
                        if (smpl_betas[i][:10] == shape_records[hased_personid]).all():
                            print(f'this is person {hased_personid}')
                            oldperson = True
                            current_person = hased_personid
                    if not oldperson:
                        person_id += 1
                        shape_records[person_id] = smpl_betas[i][:10]
                        current_person = person_id
                    oldperson = False

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

            # store data
            if not os.path.isdir(out_path+'/3dpw_vid'):
                os.makedirs(out_path+'/3dpw_vid')
            out_file = os.path.join(out_path, '3dpw_vid/{}.npz'.format(seq_name))
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
            if debug:
                # write to video
                for idx, imagename in enumerate(imgnames_):
                    imagename = os.path.join(config.PW3D_ROOT, imagename)
                    if idx ==0:
                        fps = 30
                        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
                        video_path = out_file.replace('.npz', '.avi')
                        frame = cv2.imread(imagename)
                        frame_size = (frame.shape[1], frame.shape[0])
                        videoWriter = cv2.VideoWriter(video_path,fourcc,fps,frame_size)
                        videoWriter.write(frame)
                    else:
                        frame = cv2.imread(imagename)
                        videoWriter.write(frame)
                videoWriter.release()
