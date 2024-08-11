import os
import glob

def collect_frames_and_labels(path, label):
    frames = glob.glob(os.path.join(path, '*.png'))  # Assuming frames are in .png format
    return [(frame, label) for frame in frames]

# Paths to directories
falling_paths = ['UP Fall Dataset/Falling backwards - Activity 1', 'UP Fall Dataset/Falling forward using hands - Activity 2', 'UP Fall Dataset/Falling forward using knees - Activity 3','']
normal_path = 'Picking an object - Activity 9'

# Collect frames and assign labels
falling_frames = []
for falling_path in falling_paths:
    falling_frames.extend(collect_frames_and_labels(falling_path, 1))  # Label 1 for falling

normal_frames = collect_frames_and_labels(normal_path, 0)  # Label 0 for normal activity

# Combine all frames into a single list
all_frames = falling_frames + normal_frames

# Shuffle the list to mix falling and normal frames (optional but recommended)
import random
random.shuffle(all_frames)

# Example output of all_frames
# [
#     ('path/to/falling_forward/frame1.png', 1),
#     ('path/to/normal_activity/frame2.png', 0),
#     ...
# ]
