import imageio
import cv2
from pathlib import Path as P
import argparse

vid_path = P('/home/dj/Downloads/pyslam-20220606T062132Z-001/pyslam/')

vid_name = 'colon_LOFTR_01.mp4'

full_name = str(vid_path / vid_name)

cap = cv2.VideoCapture(full_name)

images = []

fps = 15

duration = 24 # in second

# img_num = fps * duration
img_num = 200

for _ in range(img_num):
    ret, frame = cap.read()
    if ret:
        images.append(frame)
    else:
        break

gif_name = vid_name.split('.')[0] + '.gif'
gif_full_name = str(vid_path / gif_name)
print('number of images in total: %d' % len(images))
imageio.mimsave(gif_full_name, images, fps=15)