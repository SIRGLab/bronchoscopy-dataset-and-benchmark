# %%
from audioop import cross
import numpy as np
from pathlib import Path as P
import os

from scipy.fft import idst

def load_match(fname):
    pass

root_path = P('/home/dj/git/pyslam/data_prepare/colon/SyntheticColon_I/Train/processed')

loft_path = root_path / 'matches_LOFTR'
spp_path = root_path / 'matches_SUPERPOINT'

loftr_files =  list(loft_path.glob('inliers_*.txt'))

# %%
print(list(range(1,6)))
# %%
match_prev = np.loadtxt(loftr_files[0], delimiter=',')
match_cur = np.loadtxt(loftr_files[1], delimiter=',')
# %%
print(match_prev.shape)
print(match_cur.shape)
# %%
pts_prev = match_prev[:, 2:]
pts_cur = match_cur[:, :2]

# %%
from sklearn.neighbors import NearestNeighbors

# %%
metric = 'l2'
x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(pts_prev)
dist, idx = x_nn.kneighbors(pts_cur)

# %%
p0 = pts_cur[0, :]
p1 = pts_prev[5, :]

# %%
sum((p0 - p1) ** 2)
# %%
dist[0] ** 2
# %%
idx
# %%

# %%
hash_table = get_match_hash_table(dist, idx)
# %%
class crossFrameMatch():
    def __init__(self, pts_prev, pts_next, thres=None, algorithm='kd_tree', metric='l2') -> None:
        metric = 'l2'
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, \
            algorithm=algorithm, metric=metric).fit(pts_prev)
        dist, idx = x_nn.kneighbors(pts_next)
        self.hash_table = {}
        self.dist = dist
        self.idx = idx
        self.get_match_hash_table(dist, idx)
        self.prev_i = []
        self.next_i = []
        if thres is not None:
            self.filter_dist(thres)
        
    def mean_dist(self):
        dist_sum = []
        for i in self.hash_table.keys():
            dist_sum += [self.hash_table[i]['dist']]
        return sum(dist_sum) / len(dist_sum)
    
    def filter_dist(self, dist):
        for i in self.hash_table.keys():
            if self.hash_table[i]['dist'] > dist:
                self.hash_table.pop(i)
        
    def get_cross_idx(self):
        for k in self.hash_table.keys():
            next_i = self.hash_table[k]['next_i']
            self.next_i = [next_i]
            self.prev_i += [int(k)]
        
        return self.prev_i, self.next_i

    def get_match_hash_table(self, dist, idx):
        # hash_table = {}
        for next_i in range(len(idx)):
            temp_dist = dist[next_i]
            prev_i = int(idx[next_i])
            temp_dict = {
                'dist': temp_dist,
                'next_i': next_i
            }
            if self.hash_table.get(prev_i, False):
                prev_dict = self.hash_table[prev_i]
                if prev_dict['dist'] > temp_dict['dist']:
                    self.hash_table[prev_i] = temp_dict
            else:
                self.hash_table[prev_i] = temp_dict
        
        return self.hash_table
# %%
test = crossFrameMatch(pts_prev, pts_cur)
# %%
len(test.hash_table)
# %%
test.mean_dist()
# %%
# check consistent matches through all files
match_list = []
num_list = []
dist_list = []
for i in range(1, len(loftr_files)):
    pts_prev = np.loadtxt(loftr_files[i-1], delimiter=',')[:, 2:]
    pts_cur = np.loadtxt(loftr_files[i], delimiter=',')[:, :2]
    temp = crossFrameMatch(pts_prev, pts_cur)
    num_list += [len(temp.hash_table)]
    dist_list += [temp.mean_dist()]
    match_list += [temp]
    
# %%
# %%
# %%
print(num_list.mean(), dist_list.mean())
print(num_list.min(), dist_list.min())
print(num_list.max(), dist_list.max())

# %%

class Frame_LoFTR():
    def __init__(self, img, use_avg=False) -> None:
        self.cross_frame = None # save match dictionary for shared points
        self.prev_pts = None # matched points with previous frame
        self.next_pts = None # matched points with next frame
        self.shared_prev_idx = None 
        self.shared_next_idx = None
        self.shared_next_pts = None
        self.shared_prev_pts = None
        self.shared_pts = None # only used this when use_avg == True
        self.img = img 
        self.use_avg = use_avg # avg shared prev and next pts for propagation

    def set_pre_match(self, match):
        self.prev_pts = match[:, 2:]

    def set_next_match(self, match):
        self.next_pts = match[:, :2]

    def get_match_table(self, thres=None):
        self.cross_frame = crossFrameMatch(self.prev_pts, self.next_pts, thres=thres)
        self.shared_prev_idx, self.shared_next_idx = self.cross_frame.get_cross_idx()
        self.shared_next_pts = self.next_pts[self.shared_next_idx, :]
        self.shared_prev_pts = self.prev_pts[self.shared_prev_idx, :]
        if self.use_avg:
            self.shared_pts = (self.shared_next_pts + self.shared_prev_pts) / 2
        
# how to propagate between frames
def get_cross_frame_match(frame_list):
    # input list of frames, return matches between first frame to last frame
    # we could start from the second one
    # forward finding
    frame_len = len(frame_list)
    for i in range(1, frame_len-1):
        # find intersection of shared points between frames 
        cur_shared_pts = frame_list[i].shared_next_pts
        next_shared_pts = frame_list[i+1].shared_prev_pts
        temp = crossFrameMatch(cur_shared_pts, next_shared_pts, thres=0.1)
        cur_idx, next_idx = temp.get_cross_idx()
        if len(cur_idx) == 0:
            print('No matches found between frame_%02d and frame_%02d' % (0, i+1))
            break
        else:
            cur_shared_pts = frame_list[i+1].shared_next_pts[next_idx]
        


    
# %%
