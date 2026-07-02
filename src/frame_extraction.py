"""
Extracts frames from gesture videos where a hand is detected and writes them
to a new output video, effectively filtering out frames with no visible hand.

Usage:
    python frame_extraction.py --input_dir /path/to/input_gesture_class \
                                --output_dir /path/to/output_gesture_class
"""
import cv2
import argparse
import mediapipe as mp
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frames with detected hands from gesture videos."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input directory containing gesture video files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory where filtered videos will be saved.",
    )
    return parser.parse_args()


def extract_frames_with_hands(video_path, output_folder):
    """Extract only hand-detected frames from video_path and save to output_folder."""
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(5)

    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    video_filename = os.path.join(output_folder, "output_video.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            out.write(frame)

    cap.release()
    out.release()
    print(f"Extracted {frame_count} frames with hands from {video_path} and saved to {video_filename}")


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for video in os.listdir(args.input_dir):
        video_path = os.path.join(args.input_dir, video)
        extract_frames_with_hands(video_path, os.path.join(args.output_dir, video))
