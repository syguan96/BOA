import os
import cv2
import time
import glob
import random
import numpy as np
from os.path import join

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

import config
import constants
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa


IMG_DIR = config.PW3D_ROOT
class PW3D(Dataset):
    def __init__(self, options, dataset):
        super(PW3D, self).__init__()
        self.dataset = dataset
        self.options = options
        self.img_dir = IMG_DIR
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

        # load data
        annotfiles = [os.path.join(config.PW3D_ANNOT_DIR, x+'.npz') for x in constants.pw3d_annot_names]
        self.scales = []
        self.centers = []
        self.s2ds = []
        self.thetas = []
        self.betas = []
        self.imgnames = []
        self.s2ds_smpl = []
        self.genders = []
        self.totallength = 0
        self.video_flags = []

        for fileidx, annotfile in enumerate(annotfiles):
            targetscene_name = os.path.basename(annotfile)[:-4]
            data = np.load(annotfile)
            imgnames = data['imgname']
            scale = data['scale']
            center = data['center']
            theta = data['pose'].astype(np.float)
            beta = data['shape'].astype(np.float)
            s2ds_smpl = data['smpl_j2d']
            s2ds_gt = data['part']
            s2ds_openpose = np.zeros((len(imgnames), 25, 3))
            s2ds_all = np.concatenate([s2ds_openpose, s2ds_gt], axis=1)
            gender = data['gender']
            gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
            length = scale.shape[0]
            self.totallength += length
            self.scales.append(scale)
            self.centers.append(center)
            self.thetas.append(theta)
            self.betas.append(beta)
            self.imgnames.append(imgnames)
            self.s2ds_smpl.append(s2ds_smpl)
            self.s2ds.append(s2ds_all)
            self.genders.append(gender)
            vf = [fileidx]*length
            vf[-1] = -1000    # represent the end of the video
            vf[0] = -2000     # represent the start of the video
            self.video_flags += vf

        
        self.video_flags = np.stack(self.video_flags)
        assert self.video_flags.shape[0] == self.totallength
        self.scales = np.concatenate(self.scales)
        self.centers = np.concatenate(self.centers)
        self.thetas = np.concatenate(self.thetas)
        self.betas = np.concatenate(self.betas)
        self.imgnames = np.concatenate(self.imgnames)
        self.s2ds_smpl = np.concatenate(self.s2ds_smpl)
        self.s2ds = np.concatenate(self.s2ds)
        self.genders = np.concatenate(self.genders)

    def __getitem__(self, index):
        item = {}

        scale = self.scales[index].copy()
        center = self.centers[index].copy()
        s2d = self.s2ds[index].copy()
        theta = self.thetas[index].copy()
        beta = self.betas[index].copy()
        imgname = self.imgnames[index].copy()
        s2d_smpl = self.s2ds_smpl[index].copy()
        gender = self.genders[index].copy()
        video_flag = self.video_flags[index].copy()

        image = self.read_image(imgname)
        img_w, img_h = image.shape[0], image.shape[1]

        # ori image, no aug
        flip, pn, rot, sc = self.augm_params(aug=False)
        s2d, image, theta, beta, s2d_smpl = self.process_sample(image.copy(),
                                                                          theta, 
                                                                          beta, 
                                                                          s2d, 
                                                                          s2d_smpl, 
                                                                          center, 
                                                                          scale, 
                                                                          flip, pn, rot, sc, is_train=False)
        item['keypoint'] = s2d
        item['image'] = image
        item['pose'] = theta
        item['betas'] = beta
        item['smpl_j2ds'] = s2d_smpl
        item['gender'] = gender
        item['imgname'] = imgname
        item['video_flag'] = video_flag
        return item

    def __len__(self,):
        return self.totallength

    def process_sample(self, image, pose, beta, keypoints, smpl_j2ds, center, scale, flip, pn, rot, sc, is_train):
        # labeled keypoints
        kp2d = torch.from_numpy(self.j2d_processing(keypoints, center, sc*scale, rot, flip, is_train=is_train)).float()
        smpl_j2ds = torch.from_numpy(self.j2d_processing(smpl_j2ds, center, sc*scale, rot, flip, is_train=is_train)).float()
        img = self.rgb_processing(image, center, sc*scale, rot, flip, pn, is_train=is_train)
        img = torch.from_numpy(img).float()
        img = self.normalize_img(img)
        pose = torch.from_numpy(self.pose_processing(pose, rot, flip, is_train=is_train)).float()
        betas = torch.from_numpy(beta).float()
        return kp2d, img, pose, betas, smpl_j2ds

    def read_image(self, imgname):
        imgname = os.path.join(IMG_DIR, imgname)
        img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
        return img

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn, is_train):
        rgb_img = crop(rgb_img.copy(), center, scale, [constants.IMG_RES, constants.IMG_RES], rot=rot)
        if is_train and flip:
                rgb_img = flip_img(rgb_img)
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f, is_train):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
                                  [constants.IMG_RES, constants.IMG_RES], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/constants.IMG_RES - 1.
        # flip the x coordinates
        if is_train and f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def pose_processing(self, pose, r, f, is_train):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        if is_train:
            # rotation or the pose parameters
            pose[:3] = rot_aa(pose[:3], r)
            # flip the pose parameters
            if f:
                pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def augm_params(self, aug=False):
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        if aug:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
            sc = min(1+self.options.scale_factor,
                    max(1-self.options.scale_factor, np.random.randn()*self.options.scale_factor+1))
        return flip, pn, rot, sc