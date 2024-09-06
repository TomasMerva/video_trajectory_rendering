#!/usr/bin/env python
import numpy as np
import cv2


class CameraCalibration():
    def __init__(self, video, ground_truth_points) -> None:
        self._input_video = cv2.VideoCapture(video)
        self._points_selected = []
        self._points_selected_flag = False
        self._new_point_selected = False
        self._points_reference = np.asarray(ground_truth_points, dtype=np.float32)
        self.frame_name = "frame"
        cv2.namedWindow(self.frame_name)
        cv2.setMouseCallback(self.frame_name, self._detect_mouse_click)


    def _detect_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self._points_selected.append([x, y])
            print(f"Selected point number {len(self._points_selected)} at ({x},{y})")
            self._new_point_selected = True
            if len(self._points_selected) == 4:
                print("Four calibration points selected. Starting to calibrate...")
                self._points_selected_flag = True
    
    def _compute_camera_calibration(self, p_pixels, p_reference):
        A = []
        
        for i in range(p_pixels.shape[0]):
            x, y = p_pixels[i, 0], p_pixels[i, 1]
            X, Y = p_reference[i, 0], p_reference[i, 1]
            
            Ax = [-x, -y, -1, 0, 0, 0, x * X, y * X, X]
            Ay = [0, 0, 0, -x, -y, -1, x * Y, y * Y, Y]
            
            A.append(Ax)
            A.append(Ay)
        
        # Convert A to a numpy array
        A = np.array(A)
        # Perform Singular Value Decomposition (SVD)
        U, S, Vh = np.linalg.svd(A)
        
        # Homography matrix is the last column of V reshaped to 3x3
        H = Vh[-1, :].reshape(3, 3)
        
        print("\nCamera calibration matrix:\n", H)
        return H

    def _select_points(self, frame):
        print("Select 4 points by clicking in the picture...")
        while not self._points_selected_flag:
            cv2.imshow(self.frame_name, frame)
            if self._new_point_selected:
                self._new_point_selected = False
                x, y = self._points_selected[-1]
                frame = cv2.circle(frame, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        assert len(self._points_selected) == 4, "You have not selected 4 points!"
        self._points_selected = np.asarray(self._points_selected, dtype=np.float32)
    
    def _project_point(self, point):        
        # Add a row of ones to the input points (convert to homogeneous coordinates)
        ones = np.ones((1))
        point_homogeneous = np.hstack((point.T, ones))
    
        # Apply the homography transformation
        q = np.dot(np.linalg.inv(self._calibration_matrix), point_homogeneous)

        # Normalize by the third row      
        lambda_vals = q[2]
        pout = np.vstack((q[0] / lambda_vals, q[1] / lambda_vals)).T      
        return pout

    def _test_calibration(self, frame):
        print("\nPlotting reference points for testing purposes.")
        for idx, point in enumerate(self._points_reference):
            imagePoint = np.int32(self._project_point(point))

            x,y = imagePoint[0,0], imagePoint[0,1]
            frame = cv2.circle(frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
            frame = cv2.putText(frame, str(idx+1), (x-25, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(self.frame_name, frame)
            cv2.waitKey(1)
        print("Press any key to end the calibration process...")
        cv2.waitKey(0)

    def _export_calibration_matrix(self):
        output_file = './camera_calibration_matrix.csv'
        np.savetxt(output_file, self._calibration_matrix, delimiter=',')
        print(f"Calibration matrix saved as {output_file}")



    def calibrate(self):
        ret, frame = self._input_video.read()
        if not ret:
            print("Cannot open video")
            return
        
        self._select_points(frame)

        self._calibration_matrix = self._compute_camera_calibration(p_pixels=self._points_selected,
                                                                    p_reference=self._points_reference)

        self._export_calibration_matrix()
        
        self._test_calibration(frame)

        self._input_video.release()
        cv2.destroyAllWindows()
    

if __name__ == "__main__":
    input_video = "./examples/dinovas_topdown.mp4"
    points = [
        [4.883, -2.043],
        [5.087, 2.726],
        [-3.476, -2.789],
        [0.0, 0.0]
    ]

    calibration = CameraCalibration(video=input_video,
                                    ground_truth_points = points)
    calibration.calibrate()
