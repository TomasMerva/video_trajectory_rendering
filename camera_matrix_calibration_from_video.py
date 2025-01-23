import cv2
import numpy as np
import pandas as pd
import scipy.io # for matlab .mat

class CameraCalibration:
    def __init__(self, video, data_file, start_time, end_time, n_frames, delay=0):
        self._input_video = cv2.VideoCapture(video)
        self._data = self._load_mat_file(data_file)
        # self._data = pd.read_csv(data_file)
        self._start_time = start_time
        self._end_time = end_time
        self._n_frames = n_frames
        self._delay = delay
        self._points_selected = []
        self._points_reference = []
        self.frame_name = "Calibration Frame"
        self._calibration_matrix = None
        self._points_selected_count = 0

    def _load_mat_file(self, mat_file):
        mat_data = scipy.io.loadmat(mat_file)
        data_dt = 0.1
        # Generate timestamps in microseconds from data frequency
        mat_data['timestamp'] = np.arange(0, mat_data['system_state'].shape[0] * data_dt, data_dt) * 1e6
        data = {
            'timestamp': mat_data['timestamp'].flatten(),
            'x': mat_data['system_state'][:,0],
            'y': mat_data['system_state'][:,1]
        }
        return pd.DataFrame(data)

    def _detect_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._points_selected.append([x, y])
            self._points_selected_count += 1
            print(f"Point selected: ({x}, {y})")

    def _select_points(self, frame):
        self._points_selected_count = 0
        cv2.imshow(self.frame_name, frame)
        cv2.setMouseCallback(self.frame_name, self._detect_mouse_click)
        print("Click on the robot's position in the frame.")
        while self._points_selected_count < 1:
            if cv2.waitKey(1) & 0xFF == 27:  # Exit if ESC is pressed
                break
        cv2.destroyWindow(self.frame_name)

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

    def _export_calibration_matrix(self):
        output_file = './camera_calibration_matrix.csv'
        np.savetxt(output_file, self._calibration_matrix, delimiter=',')
        print(f"Calibration matrix saved as {output_file}")

    def _project_point(self, point):
        point = np.array(point).reshape(-1, 1)  # Ensure point is a column vector
        ones = np.ones((1, 1))
        point_homogeneous = np.vstack((point, ones))
        q = np.dot(np.linalg.inv(self._calibration_matrix), point_homogeneous)
        lambda_vals = q[2]
        pout = np.vstack((q[0] / lambda_vals, q[1] / lambda_vals)).T
        return pout

    def _test_calibration(self, frame):
        print("\nPlotting reference points for testing purposes.")
        for idx, point in enumerate(self._points_reference):
            imagePoint = np.int32(self._project_point(point))
            x, y = imagePoint[0, 0], imagePoint[0, 1]
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                frame = cv2.circle(frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
                frame = cv2.putText(frame, str(idx + 1), (x - 25, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                print(f"Point {x, y} is out of frame boundaries.")
            cv2.imshow(self.frame_name, frame)
            cv2.waitKey(1)
        print("Press any key to end the calibration process...")
        cv2.waitKey(0)

    def calibrate(self):
        fps = self._input_video.get(cv2.CAP_PROP_FPS)
        start_frame = int((self._start_time) * fps)
        end_frame = int((self._end_time) * fps)
        frame_indices = np.linspace(start_frame, end_frame, self._n_frames, dtype=int)

        for i in range(self._n_frames):
            self._input_video.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[i])
            ret, frame = self._input_video.read()
            if not ret:
                print("Cannot open video")
                return

            frame_timestamp = (frame_indices[i] / fps) * 1e6  # Convert frame index to timestamp in microseconds
            closest_index = (self._data['timestamp'] - self._delay * 1e6 - frame_timestamp).abs().idxmin()
            x = self._data.iloc[closest_index]['x']
            y = self._data.iloc[closest_index]['y']
            self._points_reference.append([x, y])

            print(f"Processing frame {frame_indices[i]} at timestamp {frame_timestamp} (closest CSV timestamp: {self._data.iloc[closest_index]['timestamp']} with x={x}, y={y})")

            self._select_points(frame)

        self._calibration_matrix = self._compute_camera_calibration(p_pixels=np.array(self._points_selected),
                                                                    p_reference=np.array(self._points_reference))

        self._export_calibration_matrix()

        self._test_calibration(frame)

        self._input_video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # input_video = "./data/IMG_4139.MOV"
    # input_file = "./data/ADRC_default3_trajexport.csv"
    input_video = "./examples/nobias_8.m4v"
    input_file = "./examples/data_nobias8.mat"
    start_time = 2  # start time in seconds
    end_time = 8    # end time in seconds
    n_frames = 8
    delay = 2.0  # delay in seconds

    calibration = CameraCalibration(video=input_video, data_file=input_file, start_time=start_time, end_time=end_time, n_frames=n_frames, delay=delay)
    calibration.calibrate()