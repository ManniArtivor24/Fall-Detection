import os
import cv2 as cv
import numpy as np
from tqdm import tqdm

def calculate_dense_optical_flow(frames):
    hsv = np.zeros_like(frames[0])
    hsv[..., 1] = 255

    flow_sequence = []

    for i in range(len(frames) - 1):
        prvs = cv.cvtColor(frames[i], cv.COLOR_BGR2GRAY)
        next = cv.cvtColor(frames[i + 1], cv.COLOR_BGR2GRAY)

        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        flow_sequence.append(bgr)

    return flow_sequence

def process_image_frames_dense_optical_flow(frame_folder, base_output_image_folder, base_output_numpy_folder, activity_name):
    frames = sorted(
        [os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.jpg') or f.endswith('.png')])

    if len(frames) < 6:
        print(f"Not enough frames to process in {activity_name}.")
        return

    # Create directories based on activity name
    output_image_folder = os.path.join(base_output_image_folder, f'Dense_OF_image_output_{activity_name}')
    output_numpy_folder = os.path.join(base_output_numpy_folder, f'Dense_OF_numpy_results_{activity_name}')

    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    if not os.path.exists(output_numpy_folder):
        os.makedirs(output_numpy_folder)

    for i in tqdm(range(len(frames) - 5), desc=f"Processing dense optical flow for '{activity_name}'"):
        frame_sequence = []

        for j in range(6):
            frame_path = frames[i + j]
            frame = cv.imread(frame_path)
            if frame is None:
                continue
            frame_sequence.append(frame)

        if len(frame_sequence) < 6:
            continue

        flow_sequence = calculate_dense_optical_flow(frame_sequence)

        # Save the flow images
        for k, flow_image in enumerate(flow_sequence):
            output_image_path = os.path.join(output_image_folder, f'flow_{i+k:04d}.png')
            cv.imwrite(output_image_path, flow_image)

            flow_data = flow_image.flatten()
            output_numpy_path = os.path.join(output_numpy_folder, f'flow_{i+k:04d}.npy')
            np.save(output_numpy_path, flow_data)

# Main Directory
image_base_folder = '/Users/manniartivor/PycharmProjects/Fall-Detection/UP Fall Dataset'

# Define base output folders
base_output_image_folder = 'Dense OF Image Results'
base_output_numpy_folder = 'Dense OF Numpy Results'

# Loop through each class/activity folder in the dataset directory
for activity_folder in os.listdir(image_base_folder):
    image_folder = os.path.join(image_base_folder, activity_folder)

    if os.path.isdir(image_folder):
        activity_name = activity_folder

        print(f"Processing dense optical flow for activity: {activity_name}")

        # Process images for dense optical flow with a progress bar for each class
        process_image_frames_dense_optical_flow(image_folder, base_output_image_folder, base_output_numpy_folder, activity_name)
