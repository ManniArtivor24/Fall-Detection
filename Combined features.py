import os
import numpy as np
from sklearn.preprocessing import StandardScaler

# Paths to the main directories containing class folders
keypoints_main_dir = '/Users/manniartivor/PycharmProjects/Fall-Detection/Keypoints Numpy Results'
optical_flow_main_dir = '/Users/manniartivor/PycharmProjects/Fall-Detection/Dense OF Numpy Results'

# Output directory for combined features
combined_feature_dir = 'Combined Feature List'
if not os.path.exists(combined_feature_dir):
    os.makedirs(combined_feature_dir)

# Placeholder to store combined features and labels
combined_features = []
labels = []

# Shared Mapping of folder names to numerical labels (binary classification)
label_map = {
    'Falling forward using hands - Activity 2': 1,
    'Falling backwards - Activity 1': 1,
    'Walking - Activity 6 ': 0,
    'Falling forward using knees - Activity 3': 1,
    'Falling from seated position - Activity 5 ': 1,
    'Falling sideways - Activity 4': 1,
    'Jumping - Activity 10': 0,
    'Laying Down - Activity 11': 0,
    'Sitting - Activity 8': 0,
    'Standing - Activity 7 ': 0,
    'Picking an object - Activity 9': 0,
}

# Loop through each class/activity folder in the keypoints directory
for class_folder in sorted(os.listdir(keypoints_main_dir)):
    keypoints_class_dir = os.path.join(keypoints_main_dir, f'keypoint_numpy_results_{class_folder}')
    optical_flow_class_dir = os.path.join(optical_flow_main_dir, f'Dense_OF_numpy_results_{class_folder}')

    # Ensure that there is a corresponding class folder in the optical flow directory
    if os.path.isdir(keypoints_class_dir) and os.path.isdir(optical_flow_class_dir):

        # Loop through each file in the class folder
        for keypoint_file in sorted(os.listdir(keypoints_class_dir)):
            keypoint_path = os.path.join(keypoints_class_dir, keypoint_file)
            optical_flow_file = keypoint_file.replace('keypoints', 'flow')
            optical_flow_path = os.path.join(optical_flow_class_dir, optical_flow_file)

            if os.path.exists(optical_flow_path):
                # Debug: Check if files are being loaded
                print(f"Loading keypoints: {keypoint_path}")
                print(f"Loading optical flow: {optical_flow_path}")

                # Load the keypoints and optical flow numpy arrays
                keypoints = np.load(keypoint_path)
                optical_flow = np.load(optical_flow_path)

                # Flatten the arrays if necessary
                keypoints = keypoints.flatten()
                optical_flow = optical_flow.flatten()

                # Concatenate the keypoints and optical flow features
                combined_feature = np.concatenate((keypoints, optical_flow), axis=0)

                # Append the combined features to the list
                combined_features.append(combined_feature)

                # Append the corresponding label using the existing label map
                labels.append(label_map[class_folder])

# Convert lists to numpy arrays
combined_features = np.array(combined_features)
labels = np.array(labels)

# Debug: Check if combined_features is empty
if combined_features.size == 0:
    print("Error: Combined features array is empty. Please check your input directories and files.")
else:
    print(f"Successfully loaded {combined_features.shape[0]} samples.")

# Save combined features and labels to .npy files in the "Combined Feature List" folder
combined_features_path = os.path.join(combined_feature_dir, 'combined_features.npy')
labels_path = os.path.join(combined_feature_dir, 'labels.npy')

np.save(combined_features_path, combined_features)
np.save(labels_path, labels)

print("Combined features and labels have been saved successfully.")

# Path to the directory where combined features and labels are stored
combined_feature_dir = 'Combined Feature List'

# Load the combined features and labels
combined_features_path = os.path.join(combined_feature_dir, 'combined_features.npy')
labels_path = os.path.join(combined_feature_dir, 'labels.npy')

combined_features = np.load(combined_features_path)
labels = np.load(labels_path)

# Scale the combined features using StandardScaler
if combined_features.ndim == 2:  # Check if the array is 2D
    scaler = StandardScaler()
    combined_features_scaled = scaler.fit_transform(combined_features)

    # Save the scaled combined features
    combined_features_scaled_path = os.path.join(combined_feature_dir, 'combined_features_scaled.npy')
    np.save(combined_features_scaled_path, combined_features_scaled)

    print("Scaled combined features have been saved successfully.")
else:
    print(f"Error: Expected 2D array, but got {combined_features.ndim}D array instead.")
