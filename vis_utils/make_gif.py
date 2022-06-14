# %%
import imageio
import cv2
from pathlib import Path as P
import argparse

vid_path = P('/home/dj/git/pyslam/data_prepare/lung/vo')

vid_name = 'lung_LOFTR_029.mp4'

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
# %%
# cat traj and vid
traj_path = vid_path/'traj_img'
traj_files = sorted(list(traj_path.glob('*.png')))
traj_files_str = [str(x) for x in traj_files]
# test = cv2.imread(traj_files_str[0])
# scale = images[0].shape[0] / cv2.imread(traj_files_str[0]).shape[0]
# dim_scale = [scale, scale]
# test_resize = cv2.resize(test, images[0].shape[0:2], cv2.INTER_LINEAR)
# test_resize.shape
cat_imgs = []
for i, v in enumerate(traj_files_str):
    traj_img = cv2.imread(v)
    match_img = images[i]
    traj_img = cv2.resize(traj_img, match_img.shape[0:2], cv2.INTER_LINEAR)
    final_img = cv2.hconcat((match_img, traj_img))
    cat_imgs += [final_img]


# %%
imageio.mimsave(gif_full_name, cat_imgs, fps=15)
# %%
def make_vid(imgs, save_fname, fps=15):
    out = None
    for i in imgs:
        # try:
        #     img = cv2.imread(i)
        # except:
        #     img = cv2.imread(str(i))
        if out is None:
            h, w, _ = i.shape
            size = (w, h)
            # fname = 'seq_%s.mp4' % str(seq_num).zfill(2)
            save_vid = save_fname
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(save_vid), fourcc, fps, size)
        out.write(i)
    out.release()

vid_full_name = gif_full_name.split('.')[0] + '_traj.mp4'
make_vid(cat_imgs, vid_full_name, fps=5)
# %%
