from glob import glob
import os
from turtle import position
import numpy as np
from pathlib import Path as P
import argparse
import cv2
from scipy.spatial.transform import Rotation as R

def pos2TUM(root_path, seq_num, save_path):
    pos_file = root_path / ('SavedPosition_S' + str(seq_num) + '.txt')
    quat_file = root_path / ('SavedRotationQuaternion_S' + str(seq_num) + '.txt') # in order x,y,z,w
    pos = np.loadtxt(pos_file, delimiter=' ')
    quat = np.loadtxt(quat_file, delimiter=' ')

    # test left hand
    new_pos, new_quat = left_hand_pos(pos, quat)

    ts = np.arange(len(pos)).reshape([-1, 1])
    tum_pos = np.concatenate((ts, new_pos, new_quat), axis=1)
    fname = 'colon_tum_%s.txt' % str(seq_num).zfill(2) 
    save_name = save_path / fname
    np.savetxt(save_name, tum_pos, delimiter=' ')
    

def frame2Vid(root_path, seq_num, save_path, fps=15):
    img_folder = root_path / ('Frames_S' + str(seq_num))
    rgb_imgs = sorted(glob(str(img_folder / 'FrameBuffer*.png')))
    out = None
    for i in rgb_imgs:
        img = cv2.imread(i)
        if out is None:
            h, w, _ = img.shape
            size = (w, h)
            fname = 'seq_%s.mp4' % str(seq_num).zfill(2)
            save_vid = save_path / fname
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(save_vid), fourcc, fps, size)
        out.write(img)
    out.release()

def left_hand_pos(pos, rot):
    r = R.from_quat(rot).as_matrix()

    TM = np.eye(4)
    TM[1, 1] = -1

    poses_mat = []
    positions = []
    quats = []
    for i in range(rot.shape[0]):
        ri = r[i]
        Pi = np.concatenate((ri, pos[i].reshape((3, 1))), 1)
        Pi = np.concatenate((Pi, np.array([0.0, 0.0, 0.0, 1.0]).reshape((1, 4))), 0)
        Pi_left = TM @ Pi @ TM   # Translate between left and right handed systems
        poses_mat.append(Pi_left)
        r_left = R.from_matrix(Pi_left[:3,:3])
        quats.append(r_left.as_quat())
        positions.append(Pi_left[:3,3].T)
    return np.array(positions), np.array(quats)
    

if __name__ == '__main__':
    root_path = P('/home/dj/git/pyslam/data_prepare/colon/SyntheticColon_I/Train')
    save_path = root_path / 'processed'
    save_path.mkdir(exist_ok=True)
    print('===> formating pose file')
    pos2TUM(root_path, seq_num=9, save_path=save_path)
    print('===> formating image file')
    frame2Vid(root_path, seq_num=9, save_path=save_path)


