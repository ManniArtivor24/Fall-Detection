import os
import numpy as np
from sklearn.model_selection import train_test_split

# Set the path to the directory where the features are stored
features_dir = '/Users/manniartivor/PycharmProjects/Fall-Detection/Keypoints Numpy Results'

# Initialize lists to hold data and labels
X = []
y = []

# Mapping of folder names to numerical labels (for binary classification)
label_map = {
    'falling_forward': 1,
    'falling_backward': 1,
    'walking': 0,
    'running': 0,
    'falling using knees': 1,
    'falling from seated position': 1,
    'falling sideways': 1,
    'jumping': 0,
    'laying down': 0,
    'sitting': 0,
    'standing': 0,
    'Picking an object': 0,
}

# Loop through each folder (class) in the features directory
for folder_name in os.listdir(features_dir):
    folder_path = os.path.join(features_dir, folder_name)

    # Check if it is a directory
    if os.path.isdir(folder_path):
        label = label_map[folder_name]

        # Loop through each saved feature file in the folder
        for filename in os.listdir(folder_path):
            feature_path = os.path.join(folder_path, filename)

            # Load the saved feature (e.g., NumPy array)
            feature = np.load(feature_path)

            # Append the feature to X and the label to y
            X.append(feature)
            y.append(label)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalize the feature data if needed
# (depends on the nature of the features)
X = X.astype('float32')

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
