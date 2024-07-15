import cv2
import numpy as np
import os

def calculate_optical_flow_lk(prev_frame, next_frame):
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Detect good features to track in the previous frame
    prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7,
                                          blockSize=7)

    # Calculate optical flow
    next_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_points, None, **lk_params)

    # Select good points
    good_prev = prev_points[status == 1]
    good_next = next_points[status == 1]

    return good_prev, good_next


def visualize_optical_flow_lk(prev_frame, next_frame, good_prev, good_next):
    flow_image = next_frame.copy()

    for i, (new, old) in enumerate(zip(good_next, good_prev)):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.line(flow_image, (a, b), (c, d), (0, 255, 0), 2)
        cv2.circle(flow_image, (a, b), 5, (0, 255, 0), -1)

    return flow_image


def process_image_frames_lk(frame_folder, output_image_folder, output_numpy_folder):
    frames = sorted(
        [os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.jpg') or f.endswith('.png')])

    if len(frames) < 2:
        print("Need at least two frames to calculate optical flow.")
        return

    prev_frame = cv2.imread(frames[0])

    for i in range(1, len(frames)):
        next_frame = cv2.imread(frames[i])

        if next_frame is None:
            continue

        good_prev, good_next = calculate_optical_flow_lk(prev_frame, next_frame)
        flow_image = visualize_optical_flow_lk(prev_frame, next_frame, good_prev, good_next)

        # Save visualized optical flow image
        output_image_path = os.path.join(output_image_folder, f'lk_optical_flow_{i:04d}.png')
        cv2.imwrite(output_image_path, flow_image)

        # Save raw optical flow data as numpy array
        flow_data = np.hstack((good_prev, good_next))
        output_numpy_path = os.path.join(output_numpy_folder, f'lk_optical_flow_{i:04d}.npy')
        np.save(output_numpy_path, flow_data)

        prev_frame = next_frame


# Example usage
frame_folder = 'path_to_your_frame_folder'
output_image_folder_lk = 'path_to_your_output_image_folder_lk'
output_numpy_folder_lk = 'path_to_your_output_numpy_folder_lk'

os.makedirs(output_image_folder_lk, exist_ok=True)
os.makedirs(output_numpy_folder_lk, exist_ok=True)

process_image_frames_lk(frame_folder, output_image_folder_lk, output_numpy_folder_lk)
