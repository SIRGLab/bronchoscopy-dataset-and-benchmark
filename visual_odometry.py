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
from enum import Enum
from pyquaternion import Quaternion

from feature_tracker import FeatureTrackerTypes, FeatureTrackingResult
from feature_types import FeatureDetectorTypes
from utils_geom import poseRt
from timer import TimerFps
from scipy.spatial.transform import Rotation as R
from pathlib import Path as P

class VoStage(Enum):
    NO_IMAGES_YET   = 0     # no image received 
    GOT_FIRST_IMAGE = 1     # got first image, we can proceed in a normal way (match current image with previous image)
    
kVerbose=True     
kMinNumFeature = 2000
kRansacThresholdNormalized = 0.0003  # metric threshold used for normalized image coordinates 
kRansacThresholdPixels = 0.1         # pixel threshold used for image coordinates 
kAbsoluteScaleThreshold = 1e-7       # absolute translation scale; it is also the minimum translation norm for an accepted motion 
kUseEssentialMatrixEstimation = True # using the essential matrix fitting algorithm is more robust RANSAC given five-point algorithm solver 
kRansacProb = 0.999
kUseGroundTruthScale = True 


# This class is a first start to understand the basics of inter frame feature tracking and camera pose estimation.
# It combines the simplest VO ingredients without performing any image point triangulation or 
# windowed bundle adjustment. At each step $k$, it estimates the current camera pose $C_k$ with respect to the previous one $C_{k-1}$. 
# The inter frame pose estimation returns $[R_{k-1,k},t_{k-1,k}]$ with $||t_{k-1,k}||=1$. 
# With this very basic approach, you need to use a ground truth in order to recover a correct inter-frame scale $s$ and estimate a 
# valid trajectory by composing $C_k = C_{k-1} * [R_{k-1,k}, s t_{k-1,k}]$. 
class VisualOdometry(object):
    def __init__(self, cam, groundtruth, feature_tracker):
        self.stage = VoStage.NO_IMAGES_YET
        self.cam = cam
        self.cur_image = None   # current image
        self.prev_image = None  # previous/reference image

        self.kps_ref = None  # reference keypoints 
        self.des_ref = None # refeference descriptors 
        self.kps_cur = None  # current keypoints 
        self.des_cur = None # current descriptors 

        self.cur_R = np.eye(3,3) # current rotation 
        self.cur_t = np.zeros((3,1)) # current translation 

        self.trueX, self.trueY, self.trueZ = None, None, None
        self.groundtruth = groundtruth
        
        self.feature_tracker = feature_tracker
        self.track_result = None 

        self.mask = None # mask of valida region of image input
        self.mask_match = None # mask of matched keypoints used for drawing 
        self.draw_img = None 

        self.init_history = True 
        self.poses = []              # history of poses
        self.t0_est = None           # history of estimated translations      
        self.t0_gt = None            # history of ground truth translations (if available)
        self.traj3d_est = []         # history of estimated translations centered w.r.t. first one
        self.traj3d_gt = []          # history of estimated ground truth translations centered w.r.t. first one     
        self.abs_poses = []          # history of estimated ground truth translations centered w.r.t. world

        self.num_matched_kps = None    # current number of matched keypoints  
        self.num_inliers = None        # current number of inliers 

        self.timer_verbose = True # set this to True if you want to print timings 
        self.timer_main = TimerFps('VO', is_verbose = self.timer_verbose)
        self.timer_pose_est = TimerFps('PoseEst', is_verbose = self.timer_verbose)
        self.timer_feat = TimerFps('Feature', is_verbose = self.timer_verbose)
        self.num_process_frames = 0
        self.kps_list = []
        self.success_flag = 1
        self.num_inliers_list = []
        
        # path to save matches
        detector_name = str(feature_tracker.detector_type).split('.')[-1]
        try:
            save_path = getattr(groundtruth, 'path')
            self.save_path = P(save_path) / ('matches_' + detector_name)
            self.save_path.mkdir(exist_ok=True)
        except:
            self.save_path = None

    def init_emt_pose(self):
        # rot = np.array([[ 0.13483874,  0.99054249, -0.02537896],
        # [-0.99066138,  0.13528892,  0.01693898],
        # [ 0.02021227,  0.02285792,  0.99953438]])
        # cam_rot = np.linalg.inv(rot)
        # starting_pos = self.groundtruth.data[0] # [ts, x, y, z, qx, qy, qz, qw]
        # rot = R.from_quat(starting_pos[4:])
        # rot_m = rot.as_matrix()
        # t = starting_pos[1:4].reshape([3, 1])
        # self.cur_R = rot_m
        # # self.cur_R = self.cur_R.dot(cam_rot)
        # self.cur_t = t
        pass

    def init_pose(self):
        # tum data only
        if self.groundtruth is not None:
            starting_pos = self.groundtruth.data[0] # [ts, x, y, z, qx, qy, qz, qw]
            rot = R.from_quat(starting_pos[4:])
            rot_m = rot.as_matrix()
            t = starting_pos[1:4].reshape([3, 1])
            self.cur_R = rot_m
            self.cur_t = t
        
    # get current translation scale from ground-truth if groundtruth is not None 
    def getAbsoluteScale(self, frame_id):  
        if self.groundtruth is not None and kUseGroundTruthScale:
            self.trueX, self.trueY, self.trueZ, scale = self.groundtruth.getPoseAndAbsoluteScale(frame_id)
            return scale
        else:
            self.trueX = 0 
            self.trueY = 0 
            self.trueZ = 0
            return 1

    def computeFundamentalMatrix(self, kps_ref, kps_cur):
            F, mask = cv2.findFundamentalMat(kps_ref, kps_cur, cv2.FM_RANSAC, param1=kRansacThresholdPixels, param2=kRansacProb)
            if F is None or F.shape == (1, 1):
                # no fundamental matrix found
                raise Exception('No fundamental matrix found')
            elif F.shape[0] > 3:
                # more than one matrix found, just pick the first
                F = F[0:3, 0:3]
            return np.matrix(F), mask 	

    def removeOutliersByMask(self, mask): 
        if mask is not None:    
            n = self.kpn_cur.shape[0]     
            mask_index = [ i for i,v in enumerate(mask) if v > 0]    
            self.kpn_cur = self.kpn_cur[mask_index]           
            self.kpn_ref = self.kpn_ref[mask_index]           
            if self.des_cur is not None: 
                self.des_cur = self.des_cur[mask_index]        
            if self.des_ref is not None: 
                self.des_ref = self.des_ref[mask_index]  
            if kVerbose:
                print('removed ', n-self.kpn_cur.shape[0],' outliers')                

    # fit essential matrix E with RANSAC such that:  p2.T * E * p1 = 0  where  E = [t21]x * R21
    # out: [Rrc, trc]   (with respect to 'ref' frame) 
    # N.B.1: trc is estimated up to scale (i.e. the algorithm always returns ||trc||=1, we need a scale in order to recover a translation which is coherent with previous estimated poses)
    # N.B.2: this function has problems in the following cases: [see Hartley/Zisserman Book]
    # - 'geometrical degenerate correspondences', e.g. all the observed features lie on a plane (the correct model for the correspondences is an homography) or lie on a ruled quadric 
    # - degenerate motions such a pure rotation (a sufficient parallax is required) or an infinitesimal viewpoint change (where the translation is almost zero)
    # N.B.3: the five-point algorithm (used for estimating the Essential Matrix) seems to work well in the degenerate planar cases [Five-Point Motion Estimation Made Easy, Hartley]
    # N.B.4: as reported above, in case of pure rotation, this algorithm will compute a useless fundamental matrix which cannot be decomposed to return the rotation 
    def estimatePose(self, kps_ref, kps_cur):	
        kp_ref_u = self.cam.undistort_points(kps_ref)	
        kp_cur_u = self.cam.undistort_points(kps_cur)	        
        self.kpn_ref = self.cam.unproject_points(kp_ref_u)
        self.kpn_cur = self.cam.unproject_points(kp_cur_u)
        if kUseEssentialMatrixEstimation:
            # the essential matrix algorithm is more robust since it uses the five-point algorithm solver by D. Nister (see the notes and paper above )
            E, self.mask_match = cv2.findEssentialMat(self.kpn_cur, self.kpn_ref, focal=1, pp=(0., 0.), method=cv2.RANSAC, prob=kRansacProb, threshold=kRansacThresholdNormalized)
            # print(E)
        else:
            # just for the hell of testing fundamental matrix fitting ;-) 
            F, self.mask_match = self.computeFundamentalMatrix(kp_cur_u, kp_ref_u)
            E = self.cam.K.T @ F @ self.cam.K    # E = K.T * F * K 
        #self.removeOutliersFromMask(self.mask)  # do not remove outliers, the last unmatched/outlier features can be matched and recognized as inliers in subsequent frames                          
        _, R, t, mask = cv2.recoverPose(E, self.kpn_cur, self.kpn_ref, focal=1, pp=(0., 0.))   
        print('recovered rotation matrix: \n' , R)
        print('recovered translation matrix: \n ', t)
        print('number of inliers: %d' % self.mask_match.sum())
        return R,t  # Rrc, trc (with respect to 'ref' frame) 		

    def save_matches(self, frame_id):
        # save matches to file for further consistency check
        
        base_fname = 'frame_%04d.txt' % frame_id
        all_match_fname = self.save_path / ('all_' + base_fname)
        inlier_fname = self.save_path / ('inliers_' + base_fname)
        all_match = np.concatenate((self.track_result.kps_cur_matched, self.track_result.kps_ref_matched), axis=1)
        inlier_idx = np.where(self.mask_match == 1)
        inlier_match = all_match[inlier_idx[0], :]
        print('======> saving matches to %s...' % self.save_path)
        np.savetxt(all_match_fname, all_match, delimiter=',')
        np.savetxt(inlier_fname, inlier_match, delimiter=',')
        

    def processFirstFrame(self):
        # skip when using LoFTR
        if self.feature_tracker.feature_manager.detector_type == FeatureDetectorTypes.LOFTR:
            self.draw_img = self.cur_image
        else:
        # only detect on the current image 
            self.kps_ref, self.des_ref = self.feature_tracker.detectAndCompute(self.cur_image)
            # convert from list of keypoints to an array of points 
            self.kps_ref = np.array([x.pt for x in self.kps_ref], dtype=np.float32) 
            self.draw_img = self.drawFeatureTracks(self.cur_image)
            self.kps_list += [self.kps_ref.shape[0]]

    def processFrame(self, frame_id):
        try:
        # track features 
            self.timer_feat.start()
            if self.feature_tracker.feature_manager.detector_type == FeatureDetectorTypes.LOFTR:
                self.track_result = self.feature_tracker.track_LoFTR(self.prev_image, self.cur_image, self.mask)
            else:
                self.track_result = self.feature_tracker.track(self.prev_image, self.cur_image, self.kps_ref, self.des_ref)
            self.timer_feat.refresh()
            # estimate pose 
            self.timer_pose_est.start()
        # try:
            R, t = self.estimatePose(self.track_result.kps_ref_matched, self.track_result.kps_cur_matched)  
            self.num_process_frames += 1   
            self.success_flag = 1
        except:
            R = np.eye(3)
            t = np.zeros([3,1])
            self.success_flag = 0
        self.timer_pose_est.refresh()
        # save matches to file
        try:
            self.save_matches(frame_id)
        except:
            pass
        # update keypoints history  
        try:
            self.kps_list += [self.track_result.kps_cur.shape[0]]
        except:
            self.kps_list += [0]

        if self.success_flag == 1:
            self.kps_ref = self.track_result.kps_ref
            self.kps_cur = self.track_result.kps_cur
            self.des_cur = self.track_result.des_cur 
            self.num_matched_kps = self.kpn_ref.shape[0] 
            self.num_inliers =  np.sum(self.mask_match)
            if kVerbose:        
                print('# matched points: ', self.num_matched_kps, ', # inliers: ', self.num_inliers)      
            # t is estimated up to scale (i.e. the algorithm always returns ||trc||=1, we need a scale in order to recover a translation which is coherent with the previous estimated ones)
            absolute_scale = self.getAbsoluteScale(frame_id)
            if(absolute_scale > kAbsoluteScaleThreshold):
                # compose absolute motion [Rwa,twa] with estimated relative motion [Rab,s*tab] (s is the scale extracted from the ground truth)
                # [Rwb,twb] = [Rwa,twa]*[Rab,tab] = [Rwa*Rab|twa + Rwa*tab]
                print('estimated t with norm |t|: ', np.linalg.norm(t), ' (just for sake of clarity)')
                self.cur_t = self.cur_t + absolute_scale*self.cur_R.dot(t) 
                self.cur_R = self.cur_R.dot(R)       
            # draw image         
            self.draw_img = self.drawFeatureTracks(self.cur_image) 
            # check if we have enough features to track otherwise detect new ones and start tracking from them (used for LK tracker) 
            if (self.feature_tracker.tracker_type == FeatureTrackerTypes.LK) and (self.kps_ref.shape[0] < self.feature_tracker.num_features): 
                self.kps_cur, self.des_cur = self.feature_tracker.detectAndCompute(self.cur_image)           
                self.kps_cur = np.array([x.pt for x in self.kps_cur], dtype=np.float32) # convert from list of keypoints to an array of points   
                if kVerbose:     
                    print('# new detected points: ', self.kps_cur.shape[0])                  
            self.kps_ref = self.kps_cur
            self.des_ref = self.des_cur

            self.num_inliers_list += [self.num_inliers]
        else:
            print('Failed to track pose, skipping frames')
            absolute_scale = self.getAbsoluteScale(frame_id)
            if(absolute_scale > kAbsoluteScaleThreshold):
                # compose absolute motion [Rwa,twa] with estimated relative motion [Rab,s*tab] (s is the scale extracted from the ground truth)
                # [Rwb,twb] = [Rwa,twa]*[Rab,tab] = [Rwa*Rab|twa + Rwa*tab]
                print('estimated t with norm |t|: ', np.linalg.norm(t), ' (just for sake of clarity)')
                self.cur_t = self.cur_t + absolute_scale*self.cur_R.dot(t) 
                self.cur_R = self.cur_R.dot(R)    
            self.kps_cur, self.des_cur = self.feature_tracker.detectAndCompute(self.cur_image)           
            self.kps_cur = np.array([x.pt for x in self.kps_cur], dtype=np.float32) # convert from list of keypoints to an array of points   
            self.kps_ref = self.kps_cur
            self.des_ref = self.des_cur
            try:
                self.draw_img = self.drawFeatureTracks(self.cur_image) 
            except:
                pass
            self.num_inliers_list += [0]

        self.updateHistory()           
        

    def track(self, img, frame_id):
        if kVerbose:
            print('..................................')
            print('frame: ', frame_id) 
        # convert image to gray if needed    
        if img.ndim>2:
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)             
        # check coherence of image size with camera settings 
        assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
        self.cur_image = img
        # manage and check stage 
        if(self.stage == VoStage.GOT_FIRST_IMAGE):
            self.processFrame(frame_id)
        elif(self.stage == VoStage.NO_IMAGES_YET):
            self.processFirstFrame()
            self.stage = VoStage.GOT_FIRST_IMAGE            
        self.prev_image = self.cur_image    
        # update main timer (for profiling)
        self.timer_main.refresh()  
  

    def drawFeatureTracks(self, img, reinit = False):
        draw_img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        num_outliers = 0        
        if(self.stage == VoStage.GOT_FIRST_IMAGE):            
            if reinit:
                for p1 in self.kps_cur:
                    a,b = p1.ravel()
                    cv2.circle(draw_img,(a,b),1, (0,255,0),-1)                    
            else:    
                for i,pts in enumerate(zip(self.track_result.kps_ref_matched, self.track_result.kps_cur_matched)):
                    drawAll = False # set this to true if you want to draw outliers 
                    if self.mask_match is not None:
                        if self.mask_match[i] or drawAll:
                            p1, p2 = pts 
                            a,b = p1.astype(int).ravel()
                            c,d = p2.astype(int).ravel()
                            cv2.line(draw_img, (a,b),(c,d), (0,255,0), 1)
                            cv2.circle(draw_img,(a,b),2, (0,0,255),-1)   
                    else:
                        num_outliers+=1

                        # draw outliers
                        if self.success_flag == 0:
                            p1, p2 = pts 
                            a,b = p1.astype(int).ravel()
                            cv2.circle(draw_img,(a,b),5, (255,0,0),-1)  # blue [b, g, r]
            if kVerbose:
                print('# outliers: ', num_outliers)     
        return draw_img            

    def updateHistory(self):
        if (self.init_history is True) and (self.trueX is not None):
            self.t0_est = np.array([self.cur_t[0], self.cur_t[1], self.cur_t[2]])  # starting translation 
            self.t0_gt  = np.array([self.trueX, self.trueY, self.trueZ])           # starting translation 
            self.init_history = False 
        if (self.t0_est is not None) and (self.t0_gt is not None):             
            p = [self.cur_t[0]-self.t0_est[0], self.cur_t[1]-self.t0_est[1], self.cur_t[2]-self.t0_est[2]]   # the estimated traj starts at 0
            self.traj3d_est.append(p)
            pg = [self.trueX-self.t0_gt[0], self.trueY-self.t0_gt[1], self.trueZ-self.t0_gt[2]]  # the groudtruth traj starts at 0  
            self.traj3d_gt.append(pg)     
            self.poses.append(poseRt(self.cur_R, p))   
        self.abs_poses.append(poseRt(self.cur_R, self.cur_t.reshape(3)))
    def save_pose_TUM(self, fname):
        tum_pose = [self.matrix2line(pos) for pos in self.poses]
        np.savetxt(fname, np.array(tum_pose), delimiter=' ')
        

    @staticmethod
    def matrix2line(pos_matrix):
        rot = R.from_matrix(pos_matrix[:3,:3])
        t = pos_matrix[:, 3].T
        quat = rot.as_quat() # [xyzw] ?
        t = t.reshape([-1, 1])
        quat = quat.reshape([-1, 1])
        final_line = np.concatenate((t, quat), axis=0)
        return final_line
    