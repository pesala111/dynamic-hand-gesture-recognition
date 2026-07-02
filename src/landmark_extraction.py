"""
Extracts hand landmarks from gesture videos and renders them on a blank background.
Uses MediaPipe via a local HandTrackingModule to isolate hand skeleton features
against a black background, producing cleaned landmark videos suitable for model training.

Usage:
    python landmark_extraction.py --input_dir /path/to/input_gesture_class \
                                   --output_dir /path/to/output_landmark_class
"""
import cv2
import os
import time
import argparse
import numpy as np
import HandTrackingModule as htm
import math


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract hand landmarks from gesture videos onto a blank background."
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
        help="Path to the output directory where landmark videos will be saved.",
    )
    return parser.parse_args()


def process_videos(input_directory, output_directory):
    """Process all videos in input_directory and write landmark videos to output_directory."""
    os.makedirs(output_directory, exist_ok=True)

    detector = htm.handDetector(detectionCon=1)
    pTime = 0
    seeYou = False
    fingScaleVal = 0
    widthCap, heightCap = 640, 460

    for video_file in os.listdir(input_directory):
        video_path = os.path.join(input_directory, video_file)
        print(f"Processing video: {video_file}")

        cap = cv2.VideoCapture(video_path)

        output_video_path = os.path.join(
            output_directory, os.path.splitext(video_file)[0] + '.avi'
        )
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        output_landmarks = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        while True:
            success, img = cap.read()
            if not success:
                break
            landmarks_frame = np.zeros_like(img)

            img = detector.findHands(img)
            lmList = detector.findPosition(img, draw=False)
            if len(lmList) != 0:
                seeYou = True
                for i in range(0, len(lmList) - 1):
                    x1, y1 = lmList[i][1], lmList[i][2]
                    x2, y2 = lmList[i + 1][1], lmList[i + 1][2]
                    cv2.line(landmarks_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                lenFingScale = math.hypot(x2 - x1, y2 - y1)

                cxRel, cyRel = round(cx / widthCap, 2), round((heightCap - cy) / heightCap, 2)

                Ax, Ay = lmList[0][1], lmList[0][2]
                Bx, By = lmList[5][1], lmList[5][2]
                Cx, Cy = lmList[17][1], lmList[17][2]
                lenAB = math.hypot(Bx - Ax, By - Ay)
                lenAC = math.hypot(Cx - Ax, Cy - Ay)
                lenRef = (lenAB + lenAC) / 2

                cv2.line(landmarks_frame, (Ax, Ay), (Bx, By), (50, 50, 50), 2)
                cv2.line(landmarks_frame, (Ax, Ay), (Cx, Cy), (50, 50, 50), 2)

                for lm in lmList:
                    x, y = lm[1], lm[2]
                    cv2.circle(landmarks_frame, (x, y), 3, (255, 255, 255), 2)
                    cv2.circle(landmarks_frame, (x, y), 4, (0, 0, 255), cv2.FILLED)

                fingScaleVal = round(lenFingScale / (lenRef * 2), 2)
                if fingScaleVal > 1:
                    fingScaleVal = 1

                cv2.line(landmarks_frame, (x1, y1), (x2, y2), (255 * (1 - fingScaleVal), 0, 255 * fingScaleVal), 2)
                cv2.circle(landmarks_frame, (x1, y1), 2, (0, 255, 0), cv2.FILLED)
                cv2.circle(landmarks_frame, (x2, y2), 2, (0, 255, 0), cv2.FILLED)

                if fingScaleVal < 0.15:
                    cv2.circle(landmarks_frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                elif fingScaleVal > 0.85:
                    cv2.circle(landmarks_frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
                else:
                    cv2.circle(landmarks_frame, (cx, cy), 2, (255, 255, 255), cv2.FILLED)

                print("I - Value:", fingScaleVal, "/ Pos:", cxRel, ";", cyRel, end='\r')
            else:
                print("O - Value: 0.00", "/ Pos: 0.00 ; 0.00 ", end='\r')

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            output_landmarks.write(landmarks_frame)

        cap.release()
        output_landmarks.release()

    print("\nDone. Landmark videos written to:", output_directory)


if __name__ == "__main__":
    args = parse_args()
    process_videos(args.input_dir, args.output_dir)
