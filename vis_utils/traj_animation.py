import io
import cv2
from matplotlib import projections
import numpy as np
import matplotlib.pyplot as plt
import imageio
from pathlib import Path as P
import cv2

gt_file = '/home/dj/git/pyslam/data_prepare/colon/SyntheticColon_I/Train/processed/traj_gt_S01.txt'

est_spp_file = '/home/dj/Downloads/pyslam-20220606T062132Z-001/pyslam/SUPERPOINT_traj_est_S01.txt'

est_loftr_file = '/home/dj/git/pyslam/data_prepare/colon/SyntheticColon_I/Train/processed/LOFTR_traj_est_S01.txt'

traj_gif_file = '/home/dj/git/pyslam/data_prepare/colon/SyntheticColon_I/Train/processed/traj_full.gif'
traj_vid_file = '/home/dj/git/pyslam/data_prepare/colon/SyntheticColon_I/Train/processed/traj_full.mp4'

img_folder = P('/home/dj/git/pyslam/data_prepare/colon/SyntheticColon_I/Train/processed/traj')
img_folder.mkdir(exist_ok=True)
# load trajectory

gt = np.loadtxt(gt_file, delimiter=' ')
est_spp = np.loadtxt(est_spp_file, delimiter=' ')
est_loftr = np.loadtxt(est_loftr_file, delimiter=' ')

# plot sin wave
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_zlabel("z")
ax.set_xlabel("x")
ax.set_ylabel("y")
gt_color = 'g'
spp_color = 'r'
loftr_color = 'b'   

def draw_traj_dots(traj, ax, idx, label, color):
    data = traj[:idx+1, :]
    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]
    ax.scatter(x, y, z, marker='o', c=color, label=label)
 


# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

plt_imgs = []
for i in range(len(est_loftr)):

    draw_traj_dots(gt, ax, i, 'gt', gt_color)
    draw_traj_dots(est_spp, ax, i, 'superpoint', spp_color)
    draw_traj_dots(est_loftr, ax, i, 'LoFTR', loftr_color)

    ax.legend()
    print('processing images %d' % i)

    # you can get a high-resolution image as numpy array!!
    plot_img_np = get_img_from_fig(fig)
    # plt_imgs.append(plot_img_np)
    img_fname = str(i).zfill(5) + '.png'
    full_img_fname = img_folder / img_fname
    plt.imsave(str(full_img_fname), plot_img_np)
    plt_imgs.append(str(full_img_fname))
    plt.cla()
print('total number of images: %d ' % len(plt_imgs))
# imageio.mimsave(traj_gif_file, plt_imgs, fps=15)




def frame2Vid(img_list, save_fname, fps=15):
    # img_folder = root_path / ('Frames_S' + str(seq_num))
    # rgb_imgs = sorted(glob(str(img_folder / 'FrameBuffer*.png')))
    out = None
    for i in img_list:
        img = cv2.imread(i)
        if out is None:
            h, w, _ = img.shape
            size = (w, h)
            # fname = 'seq_%s.mp4' % str(seq_num).zfill(2)
            # save_vid = save_path / fname
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(save_fname), fourcc, fps, size)
        out.write(img)
    out.release()

frame2Vid(plt_imgs, traj_vid_file, fps=15)