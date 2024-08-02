import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import tensorflow_hub as hub

# Load MoveNet model
model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet = model.signatures['serving_default']

def get_keypoints_from_movenet(image):
    # Resize and normalize image
    input_image = cv2.resize(image, (192, 192))
    input_image = input_image / 255.0
    input_image = np.expand_dims(input_image, axis=0).astype(np.float32)

    # Perform inference
    outputs = model.signatures['serving_default'](tf.constant(input_image))
    keypoints = outputs['output_0'].numpy()

    # Extract keypoints and scale them to the original image size
    keypoints = keypoints.reshape((17, 3))[:, :2]  # Keep only the (x, y) coordinates
    height, width, _ = image.shape
    keypoints[:, 0] *= width
    keypoints[:, 1] *= height

    return keypoints

def visualize_keypoints(image, keypoints):
    for x, y in keypoints:
        image = cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
    return image

def calculate_optical_flow_lk(prev_frame, next_frame, prev_points):
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    next_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_points, None, **lk_params)

    # Ensure status array is properly shaped
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

def process_image_frames_lk(frame_folder, output_image_folder, output_numpy_folder, keypoint_image_folder):
    frames = sorted([os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.jpg') or f.endswith('.png')])

    if len(frames) < 6:
        print("Not enough frames to process.")
        return

    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    if not os.path.exists(output_numpy_folder):
        os.makedirs(output_numpy_folder)

    if not os.path.exists(keypoint_image_folder):
        os.makedirs(keypoint_image_folder)

    for i in tqdm(range(len(frames) - 5), desc="Processing frames"):
        frame_sequence = []
        for j in range(6):
            frame_path = frames[i + j]
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            frame_sequence.append(frame)

        if len(frame_sequence) < 6:
            continue

        # Get keypoints using MoveNet with progress bar
        prev_keypoints = None
        for k in tqdm(range(1), desc=f"Processing keypoints for frame {i:04d}"):
            prev_keypoints = get_keypoints_from_movenet(frame_sequence[0])
        keypoints_sequence = [prev_keypoints]

        # Visualize keypoints on the first frame
        keypoint_image = visualize_keypoints(frame_sequence[0].copy(), prev_keypoints)
        keypoint_image_path = os.path.join(keypoint_image_folder, f'keypoints_{i:04d}.png')
        cv2.imwrite(keypoint_image_path, keypoint_image)

        for k in range(1, 6):
            good_prev, good_next = calculate_optical_flow_lk(frame_sequence[k - 1], frame_sequence[k], prev_keypoints.astype(np.float32))
            keypoints_sequence.append(good_next)
            prev_keypoints = good_next

        flow_image = visualize_optical_flow_lk(frame_sequence, keypoints_sequence)

        output_image_path = os.path.join(output_image_folder, f'flow_{i:04d}.png')
        cv2.imwrite(output_image_path, flow_image)

        flow_data = np.hstack([kp.flatten() for kp in keypoints_sequence])
        output_numpy_path = os.path.join(output_numpy_folder, f'flow_{i:04d}.npy')
        np.save(output_numpy_path, flow_data)

# Example usage
frame_folder = 'UP Fall Dataset/Falling backwards - Activity 1'
output_image_folder_lk = 'OF_image_output'
output_numpy_folder_lk = 'OF_numpy_results'
keypoint_image_folder = 'Keypoint_image_results'

process_image_frames_lk(frame_folder, output_image_folder_lk, output_numpy_folder_lk, keypoint_image_folder)
