import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time


# load movenet model
model = hub.load("https://www.kaggle.com/models/google/movenet/TensorFlow2/multipose-lightning/1")
movenet = model.signatures['serving_default']


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

#loop to go through each person in the frame(max 6)
def loop_through_persons(frame, keypoints_scores,edges,confidence_threshold):
    for person in keypoints_scores:
        draw_connections(frame, person,edges,confidence_threshold)
        draw_keypoints(frame, person,confidence_threshold)

# read in camera input with openCV
cap = cv2.VideoCapture("fall-23-cam0.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()


    # resize img for usage with model. Must be a factor of 32
    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 128, 256)
    input_img = tf.cast(img, dtype=tf.int32)

    # Detection
    results = movenet(input_img)
    keypoints_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))

    # call render function
    loop_through_persons(frame, keypoints_scores, EDGES, 0.2)

    cv2.imshow('MoveNet Pose Detection', frame)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Frame processed in {elapsed_time:.2f} seconds")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()