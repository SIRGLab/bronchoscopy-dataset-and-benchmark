import os
import cv2
from glob import glob
import numpy as np
import argparse
from pathlib import Path as P
from load_emt import get_emt_eval

def process_seq(root_path, save_path):
    pass

def slice_emt(emt_data, slice_start, slice_end):
    emt_part = emt_data[slice_start:slice_end, :]
    # correct translation
    emt_translation = emt_part[:, 1:4] - emt_part[0:1, 1:4]
    emt_part[:, 1:4] = emt_translation
    new_ts = np.arange(len(emt_part)).reshape([-1, 1])
    emt_part[:, 0:1] = new_ts
    return emt_part
    
def extract_img(vid_fname, save_path):
    try:
        vidcap = cv2.VideoCapture(vid_fname)
    except:
        vidcap = cv2.VideoCapture(str(vid_fname))
    success,image = vidcap.read()
    count = 0
    
    while success:
        img_name = "frame_%04d.jpg" % count
        save_fname = str(save_path / img_name)
        cv2.imwrite(save_fname, image)     # save frame as JPEG file

        success,image = vidcap.read()
        count += 1
    
def make_vid(imgs, save_fname, fps=15):
    out = None
    for i in imgs:
        try:
            img = cv2.imread(i)
        except:
            img = cv2.imread(str(i))
        if out is None:
            h, w, _ = img.shape
            size = (w, h)
            # fname = 'seq_%s.mp4' % str(seq_num).zfill(2)
            save_vid = save_fname
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(save_vid), fourcc, fps, size)
        out.write(img)
    out.release()

if __name__ == '__main__':
    # root_path = P('/home/dj/git/pyslam/data_prepare/new_lung')
    root_path = P('/home/dj/git/pyslam/data_prepare/lung')
    # root_path = P('/home/dj/git/pyslam/data_prepare/lung/')
    video_num = 29
    save_dir = root_path / 'processed'
    save_dir.mkdir(exist_ok=True)
    vo_folder = root_path / 'vo'
    vo_folder.mkdir(exist_ok=True)
    ip_fname = '0%d.avi'%video_num
    tag = ip_fname.split('.')[0]
    img_folder = save_dir / tag
    img_folder.mkdir(exist_ok=True)
    save_img = True
    if save_img:
        print('=====> extracting images')
        extract_img(root_path/ip_fname, img_folder)
    saved_files = sorted(list(img_folder.glob('*.jpg')))
    nframes = len(saved_files)
    print('=====> extracting emt data')
    syncd_emt = get_emt_eval(str(root_path/'%d.csv')%video_num, nframes)
    np.savetxt(str(save_dir/'synced_0%d.csv')%video_num, syncd_emt, delimiter=',')
    nframes = syncd_emt.shape[0]
    slice_start = 0 # in frame number
    slice_end = -90 # in frame number
    print('=====> slicing emt and images')
    emt_part = slice_emt(syncd_emt, slice_start, slice_end)
    img_part = saved_files[slice_start:slice_end]
    save_vid_fname = str(vo_folder / (tag + '.mp4'))
    save_emt_name = str(vo_folder / (tag + '.txt'))
    print('=====> saving emt and video for VO')
    np.savetxt(save_emt_name, emt_part, delimiter=' ')
    make_vid(img_part, str(save_vid_fname))