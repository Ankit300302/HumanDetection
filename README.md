# HumanDetection


Human Detection and Tracking System
This project implements a Human Detection and Tracking System using OpenCV. It leverages HOG (Histogram of Oriented Gradients) for human detection and KCF (Kernelized Correlation Filters) for object tracking. The system is designed to detect and track humans in a video stream, with adaptive detection intervals based on scene changes.

Features
Human Detection:

Uses HOG Descriptor with a pre-trained SVM model to detect humans in video frames.

Draws bounding boxes around detected humans.

Object Tracking:

Initializes KCF trackers for each detected human.

Updates trackers in subsequent frames to follow humans across the video.

Scene Change Detection:

Calculates the absolute difference between consecutive frames to detect significant scene changes.

Adjusts the detection interval dynamically based on the level of motion in the video.

Real-Time Visualization:

Displays the video stream with bounding boxes and coordinates of detected/tracked humans.

Supports keyboard input (q) to exit the application.

Coordinate Logging:

Logs the coordinates of tracked humans in real-time for further analysis.

Requirements
Python 3.x

OpenCV (opencv-python)

NumPy (numpy)

Install the required packages using:

pip install opencv-python numpy
Usage


Code Overview
Key Functions
load_video(source):

Loads a video file from the specified path.

Returns a cv2.VideoCapture object.

detect_humans(frame):

Detects humans in the frame using the HOG descriptor.

Draws bounding boxes around detected humans.

Returns the annotated frame and a list of detected human coordinates.

initialize_trackers(frame, humans):

Initializes KCF trackers for each detected human.

Returns a list of trackers.

update_trackers(frame, trackers):

Updates the position of each tracker in the current frame.

Draws bounding boxes and logs the coordinates of tracked humans.

Returns the annotated frame and a list of updated coordinates.

calculate_frame_difference(frame1, frame2):

Computes the absolute difference between two consecutive frames.

Returns a scalar value representing the level of change.

process_video(source):

Main function to process the video.

Handles human detection, tracking, and scene change detection.

Configuration
Detection Interval:

The system dynamically adjusts the detection interval based on scene changes.

Default interval: 10 frames.

If significant motion is detected, the interval is reduced to 5 frames.

If little motion is detected, the interval is increased to 20 frames.

Scene Change Threshold:

The threshold for detecting significant scene changes is set to 500000 (based on experimentation).

Adjust this value based on your video's characteristics.

Example Output
Video Display:

The video stream is displayed in real-time with bounding boxes around detected/tracked humans.

Coordinates of tracked humans are displayed on the screen.

Console Logs:

Logs the frame number and coordinates of tracked humans:


Frame 45 - Tracked human coordinates: [(120, 80, 64, 128), (300, 150, 70, 140)]
Future Enhancements

Multi-Object Tracking:

Integrate advanced tracking algorithms like DeepSORT or FairMOT for improved accuracy.

Real-Time Alerts:

Add functionality to trigger alerts when humans enter restricted areas.

Performance Optimization:

Implement GPU acceleration for faster processing.

Integration with Surveillance Systems:

Extend the system to work with IP cameras and live streams.
