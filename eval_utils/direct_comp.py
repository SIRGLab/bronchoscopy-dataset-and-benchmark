import os

root_path = '/home/dj/Downloads/pyslam-20220606T062132Z-001/pyslam/'

ref_file = root_path + 'traj_gt_S01.txt'
est_file = root_path + 'LOFTR_traj_est_S01.txt'

sys_cmd = 'evo_traj tum %s --ref=%s -p --plot_mode=xz' % (est_file, ref_file)
os.system(sys_cmd)

# cd test/data
# evo_traj kitti KITTI_00_ORB.txt KITTI_00_SPTAM.txt --ref=KITTI_00_gt.txt -p --plot_mode=xz