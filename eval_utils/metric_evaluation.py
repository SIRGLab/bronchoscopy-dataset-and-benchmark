from evo.core import metrics
from evo.tools import log
log.configure_logging(verbose=True, debug=True, silent=False)

import pprint
import numpy as np

from evo.tools import plot
import matplotlib.pyplot as plt

# temporarily override some package settings
from evo.tools.settings import SETTINGS
SETTINGS.plot_usetex = False
import copy
from evo.tools import file_interface
from evo.core import sync

# ref_file = "../test/data/freiburg1_xyz-groundtruth.txt"
# est_file = "../test/data/freiburg1_xyz-rgbdslam_drift.txt"

# traj_ref = file_interface.read_tum_trajectory_file(ref_file)
# traj_est = file_interface.read_tum_trajectory_file(est_file)
def metric_eval(ref_file, est_file, mode='rpe', use_align=False, pose_relation=metrics.PoseRelation.full_transformation, \
    max_diff=0.01, plot_mode=plot.PlotMode.xy):
    traj_ref = file_interface.read_tum_trajectory_file(ref_file)
    traj_est = file_interface.read_tum_trajectory_file(est_file)
    if mode.lower() == 'rpe':
        rpe_metrics_and_plot(traj_ref, traj_est, pose_relation, plot_mode=plot_mode)
    elif mode.lower() == 'ape':
        ape_metrics_and_plot(traj_ref, traj_est, max_diff=max_diff, use_align=use_align,\
            pose_relation=pose_relation, plot_mode=plot_mode)
    else:
        raise NotImplementedError


def ape_metrics_and_plot(traj_ref, traj_est, max_diff=0.01, \
    use_align=False, pose_relation = metrics.PoseRelation.full_transformation, plot_mode=plot.PlotMode.xy):
    
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff)

    # scales of traj is already corrected during the running with gt references

    traj_est_aligned = copy.deepcopy(traj_est)

    if use_align:

        traj_est_aligned.align(traj_ref, correct_scale=False, correct_only_scale=False)

    fig = plt.figure()
    traj_by_label = {
        "estimate (not aligned)": traj_est,
        "estimate (aligned)": traj_est_aligned,
        "reference": traj_ref
    }
    plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
    plt.show()

    # APE evaluation
    
    # use_aligned_trajectories = False
    if use_align:
        data = (traj_ref, traj_est_aligned) 
    else:
        data = (traj_ref, traj_est)

    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stats = ape_metric.get_all_statistics()

    seconds_from_start = [t - traj_est.timestamps[0] for t in traj_est.timestamps]
    fig = plt.figure()
    plot.error_array(fig.gca(), ape_metric.error, x_array=seconds_from_start,
                    statistics={s:v for s,v in ape_stats.items() if s != "sse"},
                    name="APE", title="APE w.r.t. " + ape_metric.pose_relation.value, xlabel="$t$ (s)")
    plt.show()

    
    fig = plt.figure()
    ax = plot.prepare_axis(fig, plot_mode)
    plot.traj(ax, plot_mode, traj_ref, '--', "gray", "reference")
    plot.traj_colormap(ax, traj_est_aligned if use_align else traj_est, ape_metric.error, 
                    plot_mode, min_map=ape_stats["min"], max_map=ape_stats["max"])
    ax.legend()
    plt.show()

# RPE evaluation

def rpe_metrics_and_plot(traj_ref, traj_est, pose_relation, plot_mode=plot.PlotMode.xy):
    data = (traj_ref, traj_est)
    # normal mode
    delta = 1
    delta_unit = metrics.Unit.frames

    # all pairs mode
    all_pairs = False  # activate
    rpe_metric = metrics.RPE(pose_relation=pose_relation, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)
    rpe_metric.process_data(data)
    rpe_stats = rpe_metric.get_all_statistics()
    print(rpe_stats)

    # plots 
    traj_ref_plot = copy.deepcopy(traj_ref)
    traj_est_plot = copy.deepcopy(traj_est)
    traj_ref_plot.reduce_to_ids(rpe_metric.delta_ids)
    traj_est_plot.reduce_to_ids(rpe_metric.delta_ids)
    seconds_from_start = [t - traj_est.timestamps[0] for t in traj_est.timestamps[1:]]

    fig = plt.figure()
    plot.error_array(fig.gca(), rpe_metric.error, x_array=seconds_from_start,
                    statistics={s:v for s,v in rpe_stats.items() if s != "sse"},
                    name="RPE", title="RPE w.r.t. " + rpe_metric.pose_relation.value, xlabel="$t$ (s)")
    plt.show()

def save_evo_traj(traj, fname):
    quat = traj.orientations_quat_wxyz
    pos = traj.positions_xyz
    ts = traj.timestamps.reshape([-1, 1])
    result = np.concatenate((ts, pos, quat[:,[1,2,3,0]]), axis=1)
    np.savetxt(fname, result, delimiter=' ')

def save_aligned_traj(fname_ref, fname_est, save_fname):
    traj_ref = file_interface.read_tum_trajectory_file(fname_ref)
    traj_est = file_interface.read_tum_trajectory_file(fname_est)
    traj_est_aligned_scaled = copy.deepcopy(traj_est)
    traj_est_aligned_scaled.align(traj_ref, correct_scale=True)
    save_evo_traj(traj_est_aligned_scaled, save_fname)


if __name__ == '__main__':
    root_folder = '/home/dj/git/pyslam/data_prepare/colon/SyntheticColon_I/Train/processed/'
    ref_file = root_folder + 'traj_gt_S01.txt'
    est_file = root_folder + 'LOFTR_traj_est_S01.txt'
    # for i in metrics.PoseRelation:
    #     metric_eval(ref_file, est_file, mode='rpe', pose_relation=i)
    metric_eval(ref_file, est_file, mode='rpe', pose_relation=metrics.PoseRelation.translation_part)

    import os
    eval_cmd = 'evo_traj tum %s --ref=%s -p --plot_mode=xz' % (ref_file, est_file)
    os.system(eval_cmd)
    pass