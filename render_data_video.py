#!/usr/bin/env python
import scipy.io # for matlab .mat
import numpy as np
import cv2
import time
import os

def homography2d_project(H, pin):
    n = pin.shape[1]
    
    # convert to homogeneous coordinates
    ones = np.ones((1, n))
    pin_homogeneous = np.vstack((pin, ones))
    
    # Apply the homography transformation
    q = np.dot(H, pin_homogeneous)

    # Normalize by the third row
    lambda_vals = q[2, :]
    pout = np.vstack((q[0, :] / lambda_vals, q[1, :] / lambda_vals)).T
    
    return pout

    

if __name__=="__main__":    
    # Load files
    input_video = cv2.VideoCapture('./examples/nobias_8.m4v')
    R_W_Camera = np.genfromtxt('./camera_calibration_matrix.csv', delimiter=',')
    log_data = scipy.io.loadmat('./examples/data_nobias8.mat')

    # Video settings
    VISUALIZE_FLAG = True
    num_of_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = input_video.get(cv2.CAP_PROP_FPS)
    resolution = (int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                 )
    start_idx_video = 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter('./examples/output.mp4', fourcc, fps, resolution)


    # Data
    traj_jackal = log_data["system_state"][:, 1:3]
    traj_box = log_data["system_state"][:, 7:8]
    top_traj = log_data["top_traj"]
    top_cost = log_data["top_cost"]
    traj_data_idx = 23
    traj_data_freq = 3


    # OpenCV drawing
    isClosed = False
    color_dingo1 = (255, 128, 0)
    thickness = 2

    input_video.set(cv2.CAP_PROP_POS_FRAMES, start_idx_video)
    for id_frame in range(num_of_frames):
        ret, frame = input_video.read()

        cost = top_cost[traj_data_idx,:] #[50,]
        norm_cost = (cost - min(cost)) / (max(cost) - min(cost))

        for sample in range(top_traj.shape[1]-1, -1, -1):
            world_points = top_traj[traj_data_idx, sample, 0:40, 0:2]
            image_points = homography2d_project(np.linalg.inv(R_W_Camera), 
                                                world_points.T)
            # draw
            color = np.array([51*(1-norm_cost[sample])+255*(norm_cost[sample]), 255*(1-norm_cost[sample])+128*(norm_cost[sample]), 255*(1-norm_cost[sample])], dtype=np.int32)
            frame = cv2.polylines(frame, np.int32([image_points]), 
                                isClosed, 
                                color.tolist(), 
                                thickness)
        
        if id_frame % traj_data_freq == 0:
            traj_data_idx += 1
        
        if VISUALIZE_FLAG:
            cv2.imshow('frame', frame)
            if cv2.waitKey(int(1.0/fps*1000)) & 0xFF == ord('q'):
                break
        
        # save video
        output_video.write(frame)

    output_video.release()
    input_video.release()
    cv2.destroyAllWindows()

