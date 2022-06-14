# %%
from cProfile import label
import numpy as np
from pathlib import Path as P
# from eval_utils.loftr_utils import draw_ROC, get_pts_thres
from loftr_utils import *
import matplotlib.pyplot as plt

# %%
# draw colon
root_path = P('/home/dj/git/pyslam/data_prepare/lung/vo')

loft_path = root_path / 'matches_LOFTR'
spp_path = root_path / 'matches_SUPERPOINT'

loftr_files =  list(loft_path.glob('inliers_*.txt'))
# spp_files = list(spp_path.glob('inliers_*.txt'))

thres_list = [1, 0.5, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02]
# check_pts_consistency(loftr_files, thres_list=thres_list, name='LoFTR')
loftr_num_list = get_pts_thres(loftr_files, thres_list)
# spp_num_list = get_pts_thres(spp_files, thres_list)
fig, ax = plt.subplots()
draw_ROC(loftr_num_list, thres_list, name='LoFTR', ax=ax)
# draw_ROC(spp_num_list, thres_list, name='SPP', ax=ax)
plt.show()


# %%

if __name__ == '__main__':

    root_path = P('/home/dj/git/pyslam/data_prepare/colon/SyntheticColon_I/Train/processed')

    loft_path = root_path / 'matches_LOFTR'
    spp_path = root_path / 'matches_SUPERPOINT'

    loftr_files =  list(loft_path.glob('inliers_*.txt'))
    spp_files = list(spp_path.glob('inliers_*.txt'))

    thres_list = [1, 0.5, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02]
    # check_pts_consistency(loftr_files, thres_list=thres_list, name='LoFTR')
    loftr_num_list = get_pts_thres(loftr_files, thres_list)
    spp_num_list = get_pts_thres(spp_files, thres_list)
    fig, ax = plt.subplots()
    draw_ROC(loftr_num_list, thres_list, name='LoFTR', ax=ax)
    draw_ROC(spp_num_list, thres_list, name='SPP', ax=ax)
    plt.show()

# %%
