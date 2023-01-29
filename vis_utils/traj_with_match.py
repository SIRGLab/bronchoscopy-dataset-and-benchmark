# %%

import cv2
import numpy as np
from traj_animation import draw_traj_gif
from pathlib import Path as P

root_path = P('/home/dj/git/pyslam/data_prepare/new_lung/vo')
match_vid = '/home/dj/git/pyslam/data_prepare/new_lung/vo/lung_LOFTR_061.mp4'
traj_img_folder = P('/home/dj/git/pyslam/data_prepare/new_lung/vo/061')
traj_img_folder.mkdir(exist_ok=True)
gt_file = str(root_path / 'traj_gt_S061.txt' )
est_file = str(root_path / 'LOFTR_traj_est_S061.txt')
gt = np.loadtxt(gt_file, delimiter=' ')
traj = np.loadtxt(est_file, delimiter=' ')
legend_list = ['gt', 'LOFTR']
traj_list = [gt, traj]
color_list = ['green', 'red']
traj_img_list = draw_traj_gif(traj_list, legend_list, color_list, traj_img_folder, None, save_gif=False)
# draw_traj_gif(traj_list, legend_list, color_list, img_folder, traj_gif_file)




# %%

vid_cap = cv2.VideoCapture(match_vid)
_, vid_img = vid_cap.read()
vid_cap.release()
init_traj_img = cv2.imread(str(traj_img_list[0]))


def make_combine_video(traj_img_list, vid_fname, save_fname):
    
    vid_cap = cv2.VideoCapture(str(vid_fname))
    is_ok, vid_img = vid_cap.read()
    vid_cap.release()
    if is_ok == False:
        print('fialed to load video')
        return
    vid_h, vid_w, _ = vid_img.shape
    vid_shape = [vid_h, vid_w]
    
    init_traj = cv2.imread(traj_img_list[0])
    resized_traj = cv2.resize(init_traj, vid_img.shape[:2])
    init_cat = cv2.hconcat((vid_img, resized_traj))
    h, w, _ = init_cat.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(save_fname), fourcc, 10, [w, h])

    # open again
    vid_cap = cv2.VideoCapture(str(vid_fname))
    for i in traj_img_list:
        traj_img = cv2.imread(str(i))
        traj_img = cv2.resize(traj_img, vid_shape)
        is_ok, vid_img = vid_cap.read()
        if is_ok == False:
            break
        img_cat = cv2.hconcat((vid_img, traj_img))
        out.write(img_cat)
    out.release()
    vid_cap.release()







#     pass
# %%
init_traj_img.shape
# %%
vid_img.shape
# %%
import matplotlib.pyplot as plt
plt.imshow(vid_img)
plt.show()
# %%
resized_traj = cv2.resize(init_traj_img, vid_img.shape[:2])
# %%
plt.imshow(resized_traj)
plt.show()
# %%
cat = cv2.hconcat([vid_img, resized_traj])
plt.imshow(cat)
plt.show()
# %%
save_fname = str(root_path / '061_with_traj.mp4')
make_combine_video(traj_img_list, match_vid, save_fname)
# %%
