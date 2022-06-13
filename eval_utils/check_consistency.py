from cProfile import label
import numpy as np
from pathlib import Path as P
from loftr_utils import *
from scipy.fft import idst
import matplotlib.pyplot as plt

if __name__ == '__main__':

    root_path = P('/home/dj/git/pyslam/data_prepare/colon/SyntheticColon_I/Train/processed')

    loft_path = root_path / 'matches_LOFTR'
    spp_path = root_path / 'matches_SUPERPOINT'

    loftr_files =  list(loft_path.glob('inliers_*.txt'))

    thres_list = [1, 0.5, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02]
    check_pts_consistency(loftr_files, thres_list=thres_list)

