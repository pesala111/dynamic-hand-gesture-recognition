"""
This code designed to extract hand landmarks from a gesture video and place them on a blank background.
The result is the hand landmarks set against a black background, achieved using the MediaPipe framework.
"""
import cv2
import os
import time
import numpy as np
import HandTrackingModule as htm
import math


print("EMMA_eye initialization\n")
detector = htm.handDetector(detectionCon=1)
pTime = 0
seeYou = False
fingScaleVal = 0
widthCap, heightCap = 640, 460
print("EMMA_eye running\n")

input_directory = "/home/pesala/Hand_gesture_dataset/15_CW_Rotation"
output_directory = "/home/pesala/Hand_gesture_dataset_V3/15_CW_Rotation"
os.makedirs(input_directory, exist_ok=True)
# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

for video_file in os.listdir(input_directory):
    video_path = os.path.join(input_directory, video_file)
    print(f"Processing video: {video_file}")

    cap = cv2.VideoCapture(video_path)

    # Create a video writer for the output video
    output_video_path = os.path.join(output_directory,os.path.splitext(video_file)[0] + '.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for MP4 format
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(3))  # Width of the input video
    frame_height = int(cap.get(4))  # Height of the input video
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
            # Draw lines between landmarks to visualize hand structure
            for i in range(0, len(lmList) - 1):
                x1, y1 = lmList[i][1], lmList[i][2]
                x2, y2 = lmList[i + 1][1], lmList[i + 1][2]
                cv2.line(landmarks_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Calculate distance between thumb and index finger
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            lenFingScale = math.hypot(x2 - x1, y2 - y1)

            # Calculate relative position of fingerscales' center
            cxRel, cyRel = round(cx / widthCap, 2), round((heightCap - cy) / heightCap, 2)

            # Calculate reference length / size of the hand as reference
            Ax, Ay = lmList[0][1], lmList[0][2]
            Bx, By = lmList[5][1], lmList[5][2]
            Cx, Cy = lmList[17][1], lmList[17][2]
            lenAB = math.hypot(Bx - Ax, By - Ay)
            lenAC = math.hypot(Cx - Ax, Cy - Ay)
            lenRef = (lenAB + lenAC) / 2

            lenAB = math.hypot(Bx - Ax, By - Ay)
            lenAC = math.hypot(Cx - Ax, Cy - Ay)
            lenRef = (lenAB + lenAC) / 2

            # Plot reference lines
            cv2.line(landmarks_frame, (Ax, Ay), (Bx, By), (50, 50, 50), 2)
            cv2.line(landmarks_frame, (Ax, Ay), (Cx, Cy), (50, 50, 50), 2)

            # Loop through all landmarks
            for lm in lmList:
                x, y = lm[1], lm[2]
                # Plot landmarks as circles
                cv2.circle(landmarks_frame, (x, y), 3, (255, 255, 255), 2)
                cv2.circle(landmarks_frame, (x, y), 4, (0, 0, 255), cv2.FILLED)

            # Calculate finger scale value
            fingScaleVal = round(lenFingScale / (lenRef * 2), 2)
            if fingScaleVal > 1:
                fingScaleVal = 1

            # Plot finger scale
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
            print("O - Value:  0.00", "/ Pos: 0.00 ; 0.00 ", end='\r')

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Write the frame to the output video inside the loop
        output_landmarks.write(landmarks_frame)
    #cv2.imshow("Img", landmarks_frame)


    cap.release()
    output_landmarks.release()


print("EMMA_eye ends\n")
time.sleep(3)
