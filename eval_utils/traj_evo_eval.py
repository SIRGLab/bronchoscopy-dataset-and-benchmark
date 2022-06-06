# %%
import copy
import logging
import sys

import evo.core.lie_algebra as lie
from evo.core import trajectory
from evo.tools import plot, file_interface, log

import numpy as np
import matplotlib.pyplot as plt
from evo.core import metrics 


logger = logging.getLogger("evo")
log.configure_logging(verbose=True)
root_path = '/home/dj/git/pyslam/data_prepare/colon/SyntheticColon_I/Train/processed/'

gt_file = 'traj_gt_S01.txt'
est_file = 'SUPERPOINT_traj_est_S01.txt'
traj_ref = file_interface.read_tum_trajectory_file(root_path+gt_file)
traj_est = file_interface.read_tum_trajectory_file(root_path+est_file)

# add artificial Sim(3) transformation
traj_est.transform(lie.se3(np.eye(3), np.array([0, 0, 0])))
traj_est.scale(1.0)

logger.info("\nUmeyama alignment with scaling")
traj_est_aligned_scaled = copy.deepcopy(traj_est)
traj_est_aligned_scaled.align(traj_ref, correct_only_scale=True)

fig = plt.figure(figsize=(8, 8))
plot_mode = plot.PlotMode.xz

ax = plot.prepare_axis(fig, plot_mode, subplot_arg=121)
plot.traj(ax, plot_mode, traj_ref, '--', 'gray')
plot.traj(ax, plot_mode, traj_est, '-', 'blue')
fig.axes.append(ax)
plt.title('not aligned')

ax = plot.prepare_axis(fig, plot_mode, subplot_arg=122)
plot.traj(ax, plot_mode, traj_ref, '--', 'gray')
plot.traj(ax, plot_mode, traj_est_aligned_scaled, '-', 'blue')
fig.axes.append(ax)
plt.title('$\mathrm{Sim}(3)$ alignment')

pose_relation = metrics.PoseRelation.rotation_angle_deg

# normal mode
delta = 1
delta_unit = metrics.Unit.frames

# all pairs mode
all_pairs = False  # activate

data = (traj_ref, traj_est_aligned_scaled)

rpe_metric = metrics.RPE(pose_relation=pose_relation, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)
rpe_metric.process_data(data)
rpe_stat = rpe_metric.get_statistic(metrics.StatisticsType.rmse)
print(rpe_stat)
rpe_stats = rpe_metric.get_all_statistics()
print(rpe_stats)




fig.tight_layout()
plt.show()
# %%

# save scaled traj
type(traj_est_aligned_scaled)
# %%
t = traj_est_aligned_scaled.orientations_quat_wxyz
xyz = traj_est_aligned_scaled.positions_xyz
ts = traj_est_aligned_scaled.timestamps
# %%
def save_evo_traj(traj, fname):
    quat = traj.orientations_quat_wxyz
    pos = traj.positions_xyz
    ts = traj.timestamps.reshape([-1, 1])
    result = np.concatenate((ts, pos, quat[:,[1,2,3,0]]), axis=1)
    np.savetxt(fname, result, delimiter=' ')
    
# %%
save_evo_traj(traj_est_aligned_scaled, './evo_TUM_pred_aligned.txt')
save_evo_traj(traj_ref, './evo_TUM_gt_aligned.txt')