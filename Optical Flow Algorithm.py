import os
import cv2
import numpy as np
from tqdm import tqdm


def calculate_optical_flow_lk(prev_frame, next_frame, prev_points):
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    next_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_points, None, **lk_params)

    status = status.reshape(-1)

    good_prev = prev_points[status == 1]
    good_next = next_points[status == 1]

    return good_prev, good_next


def visualize_optical_flow_lk(frames, keypoints_sequence):
    flow_image = cv2.addWeighted(frames[0], 0.5, frames[-1], 0.5, 0)

    for i in range(len(keypoints_sequence) - 1):
        good_prev = keypoints_sequence[i]
        good_next = keypoints_sequence[i + 1]
        for new, old in zip(good_next, good_prev):
            a, b = int(new[0]), int(new[1])
            c, d = int(old[0]), int(old[1])
            flow_image = cv2.line(flow_image, (a, b), (c, d), (0, 255, 0), 2)
            flow_image = cv2.circle(flow_image, (a, b), 5, (0, 0, 255), -1)
    return flow_image


def process_image_frames_lk(frame_folder, output_image_folder, output_numpy_folder):
    frames = sorted(
        [os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.jpg') or f.endswith('.png')])

    if len(frames) < 6:
        print("Not enough frames to process.")
        return

    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    if not os.path.exists(output_numpy_folder):
        os.makedirs(output_numpy_folder)

    for i in tqdm(range(len(frames) - 5), desc="Processing frames"):
        frame_sequence = []
        keypoints_sequence = []

        for j in range(6):
            frame_path = frames[i + j]
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            frame_sequence.append(frame)

            # Extract keypoints from the keypoints drawn on the image
            keypoints = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detector = cv2.SimpleBlobDetector_create()
            keypoints = detector.detect(gray)
            keypoints = np.array([kp.pt for kp in keypoints], dtype=np.float32)

            if keypoints is not None and len(keypoints) > 0:
                keypoints_sequence.append(keypoints)
            else:
                print(f"No keypoints found in frame {i + j}")
                break

        if len(keypoints_sequence) < 6:
            continue

        flow_image = visualize_optical_flow_lk(frame_sequence, keypoints_sequence)

        output_image_path = os.path.join(output_image_folder, f'flow_{i:04d}.png')
        cv2.imwrite(output_image_path, flow_image)

        flow_data = np.hstack([kp.flatten() for kp in keypoints_sequence])
        output_numpy_path = os.path.join(output_numpy_folder, f'flow_{i:04d}.npy')
        np.save(output_numpy_path, flow_data)


# Directories
keypoint_image_folder = '/Users/manniartivor/Desktop/Major Project Reserach (2)/Fall-Detection-/keypoints_image_results Activity 2 '
output_image_folder_lk = 'OF_image_output_activity 2'
output_numpy_folder_lk = 'OF_numpy_results_activity 2'

# Use keypoint images for optical flow
process_image_frames_lk(keypoint_image_folder, output_image_folder_lk, output_numpy_folder_lk)