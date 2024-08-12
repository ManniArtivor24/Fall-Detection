import cv2
import os
import matplotlib.pyplot as plt

# File path for the original video
original_video_path = '/home/ntu-user/PycharmProjects/Assesment/Video /COMP40731_video.mp4'

# Directories to save frames and audio
frames_dir = 'Frames'
os.makedirs(frames_dir, exist_ok=True)

# Open the original video capture
cap_orig = cv2.VideoCapture(original_video_path)

# Check if the original video capture is successfully opened
if not cap_orig.isOpened():
    print(f"Error: Unable to open the video at path {original_video_path}")
else:
    frame_number = 1
    while True:
        # Read frame from the original video
        ret_orig, frame_orig = cap_orig.read()
        if not ret_orig:
            break

        # Resize the frame to the target size for easier handling
        resized_frame = cv2.resize(frame_orig, (320, 240), interpolation=cv2.INTER_LINEAR)

        # Save the resized frame as an image using matplotlib to ensure RGB format
        frame_filename = os.path.join(frames_dir, f"frame{frame_number:04d}.jpg")
        plt.imsave(frame_filename, cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

        frame_number += 1

    # Release the video capture object
    cap_orig.release()

    print(f"Frames saved to: {frames_dir}")
    print(f"Number of frames processed: {frame_number - 1}")

