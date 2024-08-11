import numpy as np

# Load keypoints and optical flow data
keypoints_data = np.load('keypoints_0015.npy')  # Shape: (num_frames, num_keypoints)
optical_flow_data = np.load('flow_0015.npy')  # Shape: (num_frames, height, width, 2)

# Ensure that the dimensions match
assert keypoints_data.shape[0] == optical_flow_data.shape[0], "Mismatch in the number of frames"

num_frames = keypoints_data.shape[0]
keypoints_dim = keypoints_data.shape[1]
optical_flow_dim = optical_flow_data.shape[1] * optical_flow_data.shape[2] * optical_flow_data.shape[3]

# Flatten optical flow data
optical_flow_flat = optical_flow_data.reshape(num_frames, -1)

# Combine keypoints and optical flow features
combined_features = np.hstack((keypoints_data, optical_flow_flat))

sequence_length = 10  # Define the length of the sequences
X = []
y = []

# Generate sequences
for i in range(num_frames - sequence_length):
    X.append(combined_features[i:i + sequence_length])
    y.append(labels[i + sequence_length])  # Assuming you have labels for each frame

X = np.array(X)
y = np.array(y)

# Save the sequences to .npy files for later use
np.save('X_sequences.npy', X)
np.save('y_sequences.npy', y)

