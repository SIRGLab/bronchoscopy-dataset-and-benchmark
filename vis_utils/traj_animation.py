import io
import cv2
from matplotlib import projections
import numpy as np
import matplotlib.pyplot as plt
import imageio
from pathlib import Path as P
import cv2

# plot sin wave


def draw_traj_dots(traj, ax, idx, label, color):
    data = traj[:idx+1, :]
    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]
    ax.scatter(x, y, z, marker='o', c=color, label=label)
 


# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=100):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def draw_traj_gif(traj_list, name_list, color_list, img_folder, save_fname, end_frame=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_zlabel("z")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # gt_color = 'g'
    # spp_color = 'r'
    # loftr_color = 'b'   
    plt_imgs_fnames = []
    plt_imgs = []
    if end_frame is None:
        total_len = len(traj_list[0])
    else:
        total_len = end_frame
    for i in range(total_len):

        # draw_traj_dots(gt, ax, i, 'gt', gt_color)
        # draw_traj_dots(est_spp, ax, i, 'superpoint', spp_color)
        # draw_traj_dots(est_loftr, ax, i, 'LoFTR', loftr_color)
        for j in range(len(name_list)):
            draw_traj_dots(traj_list[j], ax, i, name_list[j], color_list[j])
        ax.legend()
        print('processing images %d' % i)

        # you can get a high-resolution image as numpy array!!
        plot_img_np = get_img_from_fig(fig)
        plt_imgs.append(plot_img_np)
        img_fname = str(i).zfill(5) + '.png'
        full_img_fname = img_folder / img_fname
        plt.imsave(str(full_img_fname), plot_img_np)
        plt_imgs_fnames.append(str(full_img_fname))

        plt.cla()
    print('total number of images: %d ' % len(plt_imgs))
    imageio.mimsave(save_fname, plt_imgs, fps=15)
    return plt_imgs_fnames



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

def vid2gif(vid_fname, gif_fname):
    
    pass

if __name__ == '__main__':

    gt_file = '/home/dj/git/pyslam/data_prepare/colon/SyntheticColon_I/Train/processed/traj_gt_S01.txt'

    est_spp_file = '/home/dj/git/pyslam/data_prepare/colon/SyntheticColon_I/Train/processed/SUPERPOINT_traj_est_S01.txt'

    est_loftr_file = '/home/dj/git/pyslam/data_prepare/colon/SyntheticColon_I/Train/processed/LOFTR_traj_est_S01.txt'

    traj_gif_file = '/home/dj/git/pyslam/data_prepare/colon/SyntheticColon_I/Train/processed/traj_full.gif'
    traj_vid_file = '/home/dj/git/pyslam/data_prepare/colon/SyntheticColon_I/Train/processed/traj_full.mp4'

    img_folder = P('/home/dj/git/pyslam/data_prepare/colon/SyntheticColon_I/Train/processed/traj')
    img_folder.mkdir(exist_ok=True)
    # load trajectory

    gt = np.loadtxt(gt_file, delimiter=' ')
    est_spp = np.loadtxt(est_spp_file, delimiter=' ')
    est_loftr = np.loadtxt(est_loftr_file, delimiter=' ')

    traj_list = [gt, est_spp, est_loftr]
    legend_list = ['gt', 'spp', 'loftr']
    color_list = ['g', 'b', 'r']
    draw_traj_gif(traj_list, legend_list, color_list, img_folder, traj_gif_file)