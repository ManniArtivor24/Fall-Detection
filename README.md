# Fall-Detection
# Overview
This project aims to develop an AI-based fall detection system for elderly care using computer vision and deep learning. By leveraging Pose Estimation (MoveNet) and Optical Flow data, two LSTM models were built to enhance detection accuracy and reduce false positives.

# Features
- Pose Estimation: Real-time detection of body joints with MoveNet.
- Optical Flow Analysis: Tracks motion dynamics to differentiate falls from normal activities.
- LSTM Models: Analyze sequential data for precise fall detection.

# Requirements
1. Python 3.x
2. TensorFlow
3. OpenCV
4. Keras
5. Tensorflow Hub
# Evaluation
- Keypoint LSTM Model: Achieved 67% accuracy, focusing on keypoint data.
- Optical Flow LSTM Model: Achieved 61% accuracy, analyzing motion dynamics.
# Future Work
+ Improve model precision and recall.
+ Expand dataset diversity for better generalization.
