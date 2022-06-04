from threading import RLock
from feature_superpoint import convert_superpts_to_keypoints

# template for Custom keypoint detector and descriptor

class templateDLFE():
    def __init__(self, do_cuda=True, kp_num=100) -> None:
        self.lock = RLock()
        self.fe = None
        self.cur_frame = None
        self.prev_frame = None
        self.keypoint_size = kp_num
        self.pts = []
        self.des = []
        self.heatmap = []
        pass


    def detectAndCompute(self, frame, mask=None):
        '''
        input:
            param: frame, current frame
            mask: mask on valid input
        return:
            keypoints,
            descriptors
        '''
        raise NotImplementedError

        


    def detect(self, frame):
        '''
        '''
        raise NotImplementedError


    def compute(self, frame, kps=None, mask=None):
        '''
        '''
        raise NotImplementedError

    def detectAndMatch(self, prev_frame, cur_frame, mask=None):
        '''
        return keypoint locations and matches directly, without descriptor
        '''
        raise NotImplementedError