"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import cv2
import math
import os

from config import Config

from visual_odometry import VisualOdometry
from camera  import PinholeCamera
from ground_truth import groundtruth_factory
from dataset import dataset_factory
from utils_geom import savePoseseFile, saveGTPoseFile
from eval_utils.metric_evaluation import save_aligned_traj
#from mplot3d import Mplot3d
#from mplot2d import Mplot2d
from mplot_thread import Mplot2d, Mplot3d

from feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from feature_manager import feature_manager_factory
from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo
from feature_matcher import feature_matcher_factory, FeatureMatcherTypes

from feature_tracker_configs import FeatureTrackerConfigs
from pathlib import Path as P
from feature_types import detector_idxing
from eval_utils.metric_evaluation import metric_eval
from evo.core import metrics
"""
use or not pangolin (if you want to use it then you need to install it by using the script install_thirdparty.sh)
"""
kUsePangolin = False  

if kUsePangolin:
    from viewer3D import Viewer3D




if __name__ == "__main__":

    config = Config()

    dataset = dataset_factory(config.dataset_settings)

    groundtruth = groundtruth_factory(config.dataset_settings)

    cam = PinholeCamera(config.cam_settings['Camera.width'], config.cam_settings['Camera.height'],
                        config.cam_settings['Camera.fx'], config.cam_settings['Camera.fy'],
                        config.cam_settings['Camera.cx'], config.cam_settings['Camera.cy'],
                        config.DistCoef, config.cam_settings['Camera.fps'])


    num_features=2000  # how many features do you want to detect and track?

    # select your tracker configuration (see the file feature_tracker_configs.py) 
    # LK_SHI_TOMASI, LK_FAST
    # SHI_TOMASI_ORB, FAST_ORB, ORB, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, SUPERPOINT, FAST_TFEAT
    # about 43.29 FPS
    # tracker_config = FeatureTrackerConfigs.LK_SHI_TOMASI
    # tracker_config['num_features'] = num_features
    
    # using superpoint as detector and descriptor
    # about 23.3 FPS
    # tracker_config = FeatureTrackerConfigs.SUPERPOINT
    # tracker_config['num_features'] = num_features


    # # using ORB2 as detector and descriptor
    # 120.36 fps
    # tracker_config = FeatureTrackerConfigs.ORB2
    # tracker_config['num_features'] = num_features

    # # using SIFT as detector and descriptor
    # 22.84 fps
    # tracker_config = FeatureTrackerConfigs.SIFT
    # tracker_config['num_features'] = num_features

    # using LoFTR as matcher
    # 5.12 FPS
    tracker_config = FeatureTrackerConfigs.LOFTR
    tracker_config['num_features'] = num_features

    feature_tracker = feature_tracker_factory(**tracker_config)

    use_gt = True

    # create visual odometry object 
    if use_gt:
        vo = VisualOdometry(cam, groundtruth, feature_tracker)
    else:
        vo = VisualOdometry(cam, None, feature_tracker)

    is_draw_traj_img = True
    traj_img_size = 800
    traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
    half_traj_img_size = int(0.5*traj_img_size)
    draw_scale = 1


    data_type = config.dataset_settings['type']
    detector_name = str(feature_tracker.detector_type).split('.')[-1]
    is_record_video = True
    if is_record_video:
        out = None
        base_path = P(config.dataset_settings['base_path'])
        name = config.dataset_settings['tag'] + '_' + config.dataset_settings.get('fname', None)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        vid_name = data_type + '_' + detector_name + '_' + name + '.mp4'
        save_vid_name = str(base_path / vid_name)
        img_size = (cam.width, cam.height)
        fps = cam.fps
        out = cv2.VideoWriter(save_vid_name, fourcc, fps, img_size)


    if (data_type == 'colon') or (data_type == 'lung') or (data_type == 'endor') or (data_type == 'all'):
        init_emt = config.dataset_settings.get('init_emt', False)
        if init_emt:
            vo.init_emt_pose()
        else:
            vo.init_pose()

    is_draw_3d = True
    if kUsePangolin:
        viewer3D = Viewer3D()
    else:
        plt3d = Mplot3d(title='3D trajectory')

    is_draw_err = True 
    err_plt = Mplot2d(xlabel='img id', ylabel='m',title='error')

    is_draw_matched_points = True 
    matched_points_plt = Mplot2d(xlabel='img id', ylabel='# matches',title='# matches')

    img_id = 0

    mask = dataset.mask

    setattr(vo, 'mask', mask)

    while dataset.isOk():

        img = dataset.getImage(img_id)

        if img is not None:
            # try:
            vo.track(img, img_id)  # main VO function 
            if(img_id > 2):	       # start drawing from the third image (when everything is initialized and flows in a normal way)

                x, y, z = vo.traj3d_est[-1]
                x_true, y_true, z_true = vo.traj3d_gt[-1]

                if is_draw_traj_img:      # draw 2D trajectory (on the plane xz)
                    draw_x, draw_y = int(draw_scale*x) + half_traj_img_size, half_traj_img_size - int(draw_scale*z)
                    true_x, true_y = int(draw_scale*x_true) + half_traj_img_size, half_traj_img_size - int(draw_scale*z_true)
                    cv2.circle(traj_img, (draw_x, draw_y), 1,(img_id*255/4540, 255-img_id*255/4540, 0), 1)   # estimated from green to blue
                    cv2.circle(traj_img, (true_x, true_y), 1,(0, 0, 255), 1)  # groundtruth in red
                    # write text on traj_img
                    cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
                    text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
                    cv2.putText(traj_img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
                    # show 		
                    cv2.imshow('Trajectory', traj_img)

                if is_draw_3d:           # draw 3d trajectory 
                    if kUsePangolin:
                        viewer3D.draw_vo(vo)   
                    else:
                        plt3d.drawTraj(vo.traj3d_gt,'ground truth',color='g',marker='.')
                        plt3d.drawTraj(vo.traj3d_est,'estimated',color='r',marker='.')
                        plt3d.refresh()

                if is_draw_err:         # draw error signals 
                    errx = [img_id, math.fabs(x_true-x)]
                    erry = [img_id, math.fabs(y_true-y)]
                    errz = [img_id, math.fabs(z_true-z)] 
                    err_plt.draw(errx,'err_x',color='g')
                    err_plt.draw(erry,'err_y',color='b')
                    err_plt.draw(errz,'err_z',color='r')
                    err_plt.refresh()    

                if is_draw_matched_points:
                    matched_kps_signal = [img_id, vo.num_matched_kps]
                    inliers_signal = [img_id, vo.num_inliers]                    
                    matched_points_plt.draw(matched_kps_signal,'# matches',color='b')
                    matched_points_plt.draw(inliers_signal,'# inliers',color='g')                    
                    matched_points_plt.refresh()                    


            # draw camera image 
            cv2.imshow('Camera', vo.draw_img)	
            # save tracked image as video 
            if is_record_video:
                out.write(vo.draw_img)
        # press 'q' to exit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        img_id += 1
    print('Successfully processed frames: %d'%vo.num_process_frames)
    print('Successfully processed rate: %.2f'%(100*vo.num_process_frames/dataset.num_frames))
    print('Average detected points: %.2f'%np.array(vo.kps_list).mean())
    avg_inliers = np.array(vo.num_inliers_list).sum()/vo.num_process_frames
    fps_log = vo.timer_feat.fps_log
    avg_fps = sum(fps_log) / (len(fps_log) - 1)
    print('Average FPS: %.2f'%avg_fps)
    print('Average inliners: %.2f' % avg_inliers)
    #print('press a key in order to exit...')
    #cv2.waitKey(0)
    # poses_fname = ''
    if out is not None:
        out.release()
    est_file = savePoseseFile(config.dataset_settings, vo.abs_poses, detector_name=detector_name)
    # save gt pose in the same folder
    try:
        gt_file = saveGTPoseFile(config.dataset_settings, groundtruth)

        est_aligned_file = str(est_file)[:-4] + '_aligned.txt'
        save_aligned_traj(gt_file, est_file, est_aligned_file)
    except:
        pass
    if is_draw_traj_img:
        print('saving map.png')
        cv2.imwrite('map.png', traj_img)
    if is_draw_3d:
        if not kUsePangolin:
            plt3d.quit()
        else: 
            viewer3D.quit()
    if is_draw_err:
        err_plt.quit()
    if is_draw_matched_points is not None:
        matched_points_plt.quit()
                
    cv2.destroyAllWindows()

    # trajectory evaluation with evo
    # metric_eval(gt_file, est_file, mode='rpe', pose_relation=metrics.PoseRelation.translation_part)
    # metric_eval(gt_file, est_file, mode='rpe', pose_relation=metrics.PoseRelation.rotation_angle_deg)
    # eval_cmd = 'evo_traj tum %s --ref=%s -p --plot_mode=xz' % (gt_file, est_aligned_file)
    # os.system(eval_cmd)
