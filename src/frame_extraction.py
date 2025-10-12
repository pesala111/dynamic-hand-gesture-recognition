"""
This code is designed to extract frames from a gesture video where a hand is detected.
"""

import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def extract_frames_with_hands(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(5)

    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0

    # Define the output video file path
    video_filename = os.path.join(output_folder, "output_video.avi")

    # Define video writer parameters
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            # Write frames with detected hands to the video
            out.write(frame)

    cap.release()
    out.release()

    print(f"Extracted {frame_count} frames with hands from {video_path} and saved to {video_filename}")

root_dir = "/home/pesala/Nvidia_Gesture_Dataset_naming_V4/02_Swiping_Up"
output_dir = "/home/pesala/Nvidia_Gesture_Dataset_naming_V5/02_Swiping_Up"

for video in os.listdir(root_dir):
    video_path = os.path.join(root_dir, video)
    extract_frames_with_hands(video_path, os.path.join(output_dir, video))

