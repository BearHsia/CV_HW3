import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping
import matplotlib.pyplot as plt

random.seed(999)
#RANSAC parameters
N_tri = 10000 # number of trials
T_dis = 5 # threshold for kicking out outliers

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    #out = None

    # for all images to be stitched:
    for idx in range(len(imgs) - 1):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints and descriptors with ORB
        img1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        kp1, des1 = orb.detectAndCompute(img1_gray,None)    # queryImage
        kp2, des2 = orb.detectAndCompute(img2_gray,None)    # trainImage
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        #matches = sorted(matches, key = lambda x:x.distance)
        #matches = matches[:int(0.8*len(matches))]

        kp1_xy = np.zeros((2,len(matches)))
        kp2_xy = np.zeros((2,len(matches)))
        for idx in range(len(matches)):
            kp1_xy[0,idx] = kp1[matches[idx].queryIdx].pt[0]
            kp1_xy[1,idx] = kp1[matches[idx].queryIdx].pt[1]
            kp2_xy[0,idx] = kp2[matches[idx].trainIdx].pt[0]
            kp2_xy[1,idx] = kp2[matches[idx].trainIdx].pt[1]

        S_kp1 = np.pad(kp1_xy,((0,1),(0,0)),constant_values=1)
        S_kp2 = np.pad(kp2_xy,((0,1),(0,0)),constant_values=1)

        # TODO: 2. apply RANSAC to choose best H
        NH = np.zeros((N_tri,3,3))
        NI = np.zeros((N_tri,))         #inliners number
        ID = list(range(len(matches)))
        for trial_i in range(N_tri):
            ss = random.sample(ID,4)
            v = np.transpose(S_kp1[:2,ss])
            u = np.transpose(S_kp2[:2,ss])
            #print(v.shape)
            H_temp = solve_homography(u, v)
            est_S_kp1 = np.matmul(H_temp,S_kp2)
            est_S_kp1[0,:] = est_S_kp1[0,:]/est_S_kp1[2,:]
            est_S_kp1[1,:] = est_S_kp1[1,:]/est_S_kp1[2,:]
            est_S_kp1[2,:] = est_S_kp1[2,:]/est_S_kp1[2,:]
            dist = np.linalg.norm(est_S_kp1[:2,:]-S_kp1[:2,:],axis=0,ord=2)
            NI[trial_i] = np.sum(dist<T_dis)
            NH[trial_i,...] = H_temp
        #print(np.argmax(NI))
        H_best = NH[np.argmax(NI),...]
        #print(H_best.shape)
        # TODO: 3. chain the homographies
        last_best_H = np.matmul(last_best_H,H_best)
        # TODO: 4. apply warping
        dst = warping(im2, dst, last_best_H, 0, h_max, 0, w_max, direction='b',alpha_blending=False)
        #dst = warping(im2, dst, last_best_H, 0, h_max, 0, w_max, direction='b',alpha_blending=True,alpha=0.5)
    
    return dst


if __name__ == "__main__":

    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)