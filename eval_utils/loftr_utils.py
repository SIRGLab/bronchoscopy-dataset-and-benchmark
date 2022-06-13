from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np

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
        
        new_table = {}
        for i in self.hash_table.keys():
            if self.hash_table[i]['dist'] < dist:
                new_table[i] = self.hash_table[i]

        self.hash_table = new_table
        
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
    index_list = []
    for i in range(1, frame_len-1):
        # find intersection of shared points between frames 
        cur_shared_pts = frame_list[i].shared_next_pts
        next_shared_pts = frame_list[i+1].shared_prev_pts
        temp = crossFrameMatch(cur_shared_pts, next_shared_pts, thres=0.1)
        cur_idx, next_idx = temp.get_cross_idx()
        if len(index_list) == 0:
            index_list.append(cur_idx)
        index_list.append(next_idx)
        if len(cur_idx) == 0:
            print('No matches found between frame_%02d and frame_%02d' % (0, i+1))
            break
        else:
            cur_shared_pts = frame_list[i+1].shared_next_pts[next_idx]

def filter_dist(hash_table, dist):
        
    new_table = {}
    for i in hash_table.keys():
        if hash_table[i]['dist'] < dist:
            new_table[i] = hash_table[i]

    return new_table


def get_filter_num(match_list, thres=None):
    num_list = []
    for m in match_list:
        if thres is None:
            num_list += [len(m.hash_table)]
        else:
            temp = filter_dist(m.hash_table, thres)
            num_list += [len(temp)]
    return np.array(num_list)


def print_match(match_num, thres):
    base_str = 'number of cross frame match within %f pix distance:' %thres
    stat_list = ['mean', 'min', 'max']
    num_list = [match_num.mean(), match_num.min(), match_num.max()]
    for i in range(len(stat_list)):
        mode = stat_list[i]
        num = num_list[i]
        print('%s %s %f' %(mode, base_str, num))


def draw_ROC(num_list_thres, thres_list):
    max_list = [i.max() for i in num_list_thres]
    min_list = [i.min() for i in num_list_thres]
    mean_list = [i.mean() for i in num_list_thres]
    fig, ax = plt.subplots()
    ax.plot(max_list, '-o', label='max match')
    ax.plot(min_list, '-v', label='min match')
    ax.plot(mean_list, '-s', label='avg match')
    ax.legend()
    ax.set_title('# matches w.r.t pixel distance')
    ax.set_xlabel('pixel distance')
    ax.set_ylabel('# of match')
    plt.xticks(ticks=list(range(len(thres_list))), labels=thres_list)
    plt.show()

def check_pts_consistency(file_list, thres_list=None):
    pass
    match_list = []
    num_list = []
    dist_list = []
    for i in range(1, len(file_list)):
        pts_prev = np.loadtxt(file_list[i-1], delimiter=',')[:, 2:]
        pts_cur = np.loadtxt(file_list[i], delimiter=',')[:, :2]
        temp = crossFrameMatch(pts_prev, pts_cur)
        num_list += [len(temp.hash_table)]
        dist_list += [temp.mean_dist()]
        match_list += [temp]

    if thres_list is None:
        thres_list = [1, 0.5, 0.2, 0.1, 0.07, 0.05, 0.02]

    num_list_thres = []

    for thres in thres_list:
        num_list_thres += [get_filter_num(match_list, thres)]

    draw_ROC(num_list_thres, thres_list)