import os
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set the path to the directory where the features are stored
features_dir = '/Users/manniartivor/PycharmProjects/Fall-Detection/Dense OF Numpy Results '

# Initialize lists to hold data and labels
X = []
y = []

# Mapping of folder names to numerical labels (for binary classification)
label_map = {
    'Dense_OF_numpy_results_Falling forward using hands - Activity 2': 1,
    'Dense_OF_numpy_results_Falling backwards - Activity 1': 1,
    'Dense_OF_numpy_results_Walking - Activity 6 ': 0,
    'Dense_OF_numpy_results_Falling forward using knees - Activity 3': 1,
    'Dense_OF_numpy_results_Falling from seated position - Activity 5 ': 1,
    'Dense_OF_numpy_results_Falling sideways - Activity 4': 1,
    'Dense_OF_numpy_results_Jumping - Activity 10': 0,
    'Dense_OF_numpy_results_Laying Down - Activity 11': 0,
    'Dense_OF_numpy_results_Sitting - Activity 8': 0,
    'Dense_OF_numpy_results_Standing - Activity 7 ': 0,
    'Dense_OF_numpy_results_Picking an object - Activity 9': 0,
}


# Function to extract sequences from frames
def extract_sequences(features, labels, num_frames):
    sequences = []
    sequence_labels = []

    # Ensure there are enough frames to form at least one sequence
    if len(features) < num_frames:
        raise ValueError("Not enough frames to create sequences.")

    for i in range(len(features) - num_frames + 1):
        sequence = features[i:i + num_frames]
        sequences.append(sequence)
        sequence_labels.append(labels[i + num_frames - 1])  # Use the label of the last frame in the sequence

    return np.array(sequences), np.array(sequence_labels)


# Loop through each folder (class) in the features directory
for folder_name in os.listdir(features_dir):
    folder_path = os.path.join(features_dir, folder_name)

    # Check if it is a directory
    if os.path.isdir(folder_path):
        label = label_map[folder_name]

        # Load features from all files in the folder
        features = []
        for filename in os.listdir(folder_path):
            feature_path = os.path.join(folder_path, filename)

            # Load the saved feature (e.g., NumPy array)
            feature = np.load(feature_path)

            features.append(feature)

        features = np.array(features)

        # Extract sequences from the loaded features
        num_frames = 4  # Number of frames per sequence
        sequences, sequence_labels = extract_sequences(features, [label] * len(features), num_frames)

        # Append sequences and labels to lists
        X.extend(sequences)
        y.extend(sequence_labels)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalize the feature data if needed
# (depends on the nature of the features)
X = X.astype('float32')

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the prepared data for LSTM model
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


# Load prepared data
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Convert labels to categorical (one-hot encoding) if needed
num_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], -1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], -1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(64))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)

# Evaluate the model
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"LSTM Accuracy for Dense Optical Flow: {accuracy:.2f}")

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix for Dense Optical Flow:")
print(conf_matrix)

# Print classification report
class_report = classification_report(y_true, y_pred, target_names=[f'Class {i}' for i in range(num_classes)])
print("Classification Report for Dense Optical Flow:")
print(class_report)
