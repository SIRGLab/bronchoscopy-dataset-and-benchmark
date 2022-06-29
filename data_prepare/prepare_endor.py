import os
import h5py
from pathlib import Path
import cv2
import argparse
import numpy as np
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/dj/Documents/endors')
parser.add_argument('--save_path', type=str, default='/home/dj/Documents/endors/processed')
args = parser.parse_args()    

def make_h5_video(img_arr, video_name):
    video_name = str(video_name)
    out = None
    for i in range(img_arr.shape[0]):
        img = img_arr[i, :, :, :]
        if out is None:
            h, w, _ = img.shape
            size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_name, fourcc, 15, size)
        out.write(img)
    out.release()

def save_tum_pose(pos_fname, save_fname):

    shutil.copy(str(pos_fname), str(save_fname))

if __name__ == '__main__':
    root_path = Path(args.root_path) / 'bag_1'
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True)
    seq_list = list(root_path.iterdir())
    for i, seq in enumerate(seq_list):
        print(seq)
        # 1. make video
        # 2. save pose to TUM format
        # 3. save mask
        # 4. save intrinsics
        hdf5_file = list(seq.glob('*.hdf5'))
        try:
            data = h5py.File(str(hdf5_file[0]))
        except:
            print(hdf5_file)
            print('failed to load h5 file in folder %s' % seq)
            continue
        
        # seq_name = seq.stem
        seq_name = str(i).zfill(3)
        seq_save_name = save_path / seq_name
        seq_save_name.mkdir(exist_ok=True)
        vid_name = seq_save_name / 'video_rgb.mp4'
        mask_name = seq_save_name / 'mask.png'
        pos_name = seq_save_name / 'pose.txt'
        make_h5_video(data['color'], vid_name)
        src_pos_fname = str(seq/'stamped_groundtruth.txt')
        save_tum_pose(src_pos_fname, pos_name)
        mask = np.array(data['mask'])[0,:,:,:] * 255
        cv2.imwrite(str(mask_name), mask)
        cam_int = data['intrinsics']
        
        



