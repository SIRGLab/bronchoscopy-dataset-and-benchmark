from audioop import tostereo
from math import degrees
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from filterpy.kalman import FixedLagSmoother
from scipy.spatial.transform import Rotation as R
import os

def moving_average(x, w):
    x_list = list(x)
    # same pad
    pad_num = int(w/2.0)
    front_pad = [x[0]] * pad_num
    end_pad = [x[-1]] * pad_num
    x_padded = front_pad + x_list + end_pad
    return np.convolve(x_padded, np.ones(w), 'valid') / w

def correct_rotation(emt_seq):
    quat = emt_seq[4:]

    pass

def load_emt_translation(fname, is_smooth=True, w=5):
    
    frame = pd.read_csv(fname, sep=',')
    state = frame['State']
    ts = frame['Frame'].to_numpy(dtype=int)
    ts = ts - ts[0]
    
    pos = frame[['Tx', 'Ty', 'Tz']]
    pos_np = pos.to_numpy()
    state_np = state.to_numpy(dtype=str)
    OK_idx = np.where(state_np == 'OK')
    ts_OK = ts[OK_idx]
    ts_OK = ts_OK.reshape([-1,1])
    pos_OK = pos_np[OK_idx]

    if is_smooth:
        pos_OK[:, 0] = moving_average(pos_OK[:, 0], w)
        pos_OK[:, 1] = moving_average(pos_OK[:, 1], w)
        pos_OK[:, 2] = moving_average(pos_OK[:, 2], w)

    data = np.concatenate((ts_OK,pos_OK), axis=1)
    return data

def rot2quat(rot):

    result = np.zeros([rot.shape[0], 4])
    for i in range(rot.shape[0]):
        rotvet = rot[i,:]
        try:
            # available with scipy >= 1.7.0
            r = R.from_rotvec(rotvec=rotvet, degrees=True)
        except:
            # convert degrees to radius
            rotvet = rotvet / 180 * np.pi
            r = R.from_rotvec(rotvet)
        quat = r.as_quat()
        result[i] = quat
    return result
def find_nearest_idx(v, arr):
    idx = np.abs(arr-v).argmin()
    return idx


def get_emt_eval(fname, nframe, is_smooth=False, w=5):
    frame = pd.read_csv(fname, sep=',')
    state = frame['State']
    ts = frame['Frame'].to_numpy(dtype=int)
    

    pos = frame[['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']]
    pos_np = pos.to_numpy()

    # # reverse y axis value, convert it to camera coordinate
    # pos_np[:, 1] = -pos_np[:, 1]
    # pos_np[:, 4] = -pos_np[:, 4]
    state_np = state.to_numpy(dtype=str)
    OK_idx = np.where(state_np == 'OK')
    ts_OK = ts[OK_idx]
    ts_OK = ts_OK.reshape([-1,1])
    pos_OK = pos_np[OK_idx]

    rot_vec = pos_OK[:,3:]

    


    quat_vec = rot2quat(rot_vec)

    ts_OK = ts_OK - ts_OK[0]
    ts_OK = ts_OK / max(ts_OK) * (nframe-1)
    img_ts = np.arange(nframe)
    idx_list = [find_nearest_idx(x, ts_OK) for x in img_ts]

    if is_smooth:
        pos_OK[:, 0] = moving_average(pos_OK[:, 0], w)
        pos_OK[:, 1] = moving_average(pos_OK[:, 1], w)
        pos_OK[:, 2] = moving_average(pos_OK[:, 2], w)


    # ts_sync = ts_OK[idx_list] 
    pos_sync = pos_OK[idx_list]
    quat_sync = quat_vec[idx_list]
    img_ts = img_ts.reshape([-1, 1])

    data = np.concatenate((img_ts,pos_sync[:,:3],quat_sync), axis=1)
    return data

if __name__ == '__main__':
    fname = './exp1_029.csv'
    img_list = os.listdir('./img/029/')
    nframe = len(img_list)

    t = get_emt_eval(fname, nframe)
    np.savetxt('./emt_029_eval.txt', t, delimiter=',')
    print('success!')
    # frame = pd.read_csv(fname, sep=',')
    # partial_frame = frame[['Frame', 'Rz', 'Ry', 'Rx', 'Tx', 'Ty', 'Tz']]
    # state = frame['State']
    # pos = frame[['Tx', 'Ty', 'Tz']]
    # pos_np = pos.to_numpy()
    # state_np = state.to_numpy(dtype=str)
    # OK_idx = np.where(state_np == 'OK')
    # fail_idx = np.where(state_np != 'OK')
    # pos_OK = pos_np[OK_idx]
    # fls = FixedLagSmoother(dim_x=3, dim_z=3)
    # fls.x = pos_OK[0:1].T
    # xhatsmooth, xhat = fls.smooth_batch(np.expand_dims(pos_OK, axis=-1), N=10)

    # smooth_x = xhatsmooth[:,0,0]
    # smooth_y = xhatsmooth[:,1,0]
    # smooth_z = xhatsmooth[:,2,0]

    # ax = plt.axes(projection='3d')
    # ax.plot3D(smooth_x, smooth_y, smooth_z, 'blue')

    # ax.plot3D(xhat[:,0,0], xhat[:,1,0], xhat[:,2,0], 'green')
    # def moving_average(x, w):
    #     return np.convolve(x, np.ones(w), 'valid') / w
    # ma4_x = moving_average(xhat[:,0,0], 4)
    # ma4_y = moving_average(xhat[:,1,0], 4)
    # ma4_z = moving_average(xhat[:,2,0], 4)
    # ax.plot3D(ma4_x, ma4_y, ma4_z, 'red')

    # plt.show()

    # concat and export

    
