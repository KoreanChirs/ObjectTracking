import numpy as np
from filterpy.kalman import KalmanFilter


def create_kalman_filter(dt, state_var, measure_var, state_dim, measure_dim):
    kf = KalmanFilter(dim_x=state_dim, dim_z=measure_dim)
    
    # state vector
    kf.x = np.zeros((state_dim, 1))
    
    # state transition matrix
    kf.F = np.array([[1, 0, 0, 0, dt, 0, 0, 0],
                     [0, 1, 0, 0, 0, dt, 0, 0],
                     [0, 0, 1, 0, 0, 0, dt, 0],
                     [0, 0, 0, 1, 0, 0, 0, dt],
                     [0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1]])
    
    # observation matrix
    kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0], # x can be measured
                     [0, 1, 0, 0, 0, 0, 0, 0], # y can be measured
                     [0, 0, 1, 0, 0, 0, 0, 0], # w can be measured
                     [0, 0, 0, 1, 0, 0, 0, 0]]) # h can be measured
    
    # process noise covariance matrix
    # system's uncertainty -> state uncertainty
    kf.Q = np.eye(state_dim) * state_var
    
    # measurement noise convariance matrix
    # measurement's uncertainty
    kf.R = np.eye(measure_dim) * measure_var
    
    # error convariance matrix
    # initial state's uncertainty
    kf.P = np.eye(state_dim)
    
    return kf

def estimate_next_roi(kf: KalmanFilter, rois):
    match len(rois):
        case 0:
            recent_rois = []
        case 1:
            recent_rois = rois[-1]
        case 2:
            recent_rois = rois[-2:]
        case 3:
            recent_rois = rois[-3:]
        case _:
            recent_rois = rois[-4:]
    
    # setup initial state
    if not len(recent_rois):
        return [None, None, None, None]
    
    initial_roi = recent_rois[0]
    x, y, w, h = initial_roi
    kf.x = np.array([[x], [y], [w], [h], [0], [0], [0], [0]])
    
    for roi in recent_rois:
        x, y, w, h = roi
        z = np.array([[x], [y], [w], [h]])
        kf.predict()
        kf.update(z)
    
    kf.predict()
    next_state = kf.x
    next_roi = (next_state[0, 0], next_state[1, 0], next_state[2, 0], next_state[3, 0])
    
    return np.array(next_roi)


if __name__ == '__main__':
    kf = create_kalman_filter(dt=1, state_var=1e-4, measure_var=1e-1, state_dim=8, measure_dim=4)
    
    # example
    rois = [(90, 140, 48, 48), 
            (95, 145, 49, 49), 
            (100, 150, 50, 50), 
            (102, 152, 51, 51), 
            (104, 154, 52, 52), 
            (106, 156, 53, 53)]
    
    # Estimate the next ROI using the most recent 4 data points
    next_roi = estimate_next_roi(kf, rois)
    print("Next ROI:", next_roi)
