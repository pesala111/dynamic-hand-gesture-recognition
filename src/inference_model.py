"""
Real-time gesture inference using a trained ResNet-3D model and MediaPipe hand tracking.
Preprocesses a gesture video by extracting hand landmark frames, then runs the model
to predict the gesture class.

Usage:
    python inference_model.py --model_path gesture_model.pth \
                               --video_path /path/to/gesture_video.avi \
                               --num_classes 17
"""
import cv2
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torchvision.models.video import R3D_18_Weights
import HandTrackingModule as htm
import math


# Gesture classes in sorted order (must match training label assignment)
GESTURE_CLASSES = [
    '01_Horizontal_swiping',
    '02_Swiping_Up',
    '03_Swiping_Down',
    '04_V_Swiping_Left',
    '05_V_Swiping_Right',
    '06_V_Swiping_Up',
    '07_V_Swiping_Down',
    '08_Pointing',
    '09_Pulling',
    '10_Palm_Opening',
    '11_Palm_Shake',
    '12_Peace_Sign',
    '13_Three_Finger_Open',
    '14_Pushing',
    '15_CW_Rotation',
    '16_CCW_Rotation',
    '17_Five_Finger_Closure',
]

# Build label_to_int from the sorted class list (matches recognition_model.py)
LABEL_TO_INT = {cls: idx for idx, cls in enumerate(GESTURE_CLASSES)}
INT_TO_LABEL = {idx: cls for cls, idx in LABEL_TO_INT.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Run gesture inference on a video file.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model weights (.pth file).",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to the input gesture video file.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=17,
        help="Number of gesture classes the model was trained on (default: 17).",
    )
    return parser.parse_args()


class ResNet3D(nn.Module):
    """ResNet-3D model with Dropout classification head (matches training architecture)."""

    def __init__(self, num_classes, dropout_prob=0.5):
        super(ResNet3D, self).__init__()
        self.resnet3d = models.video.r3d_18(weights=R3D_18_Weights.DEFAULT)
        num_features = self.resnet3d.fc.in_features
        self.resnet3d.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x):
        return self.resnet3d(x)


def load_model(model_path, num_classes=17):
    """Load trained ResNet3D weights from model_path."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet3D(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def preprocess_video(video_path):
    """Extract landmark frames from video_path and return a normalised tensor."""
    detector = htm.HandDetector(detectionCon=1)

    fingScaleVal = 0
    widthCap, heightCap = 640, 460
    pTime = 0

    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        landmarks_frame = np.zeros_like(img)

        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            for i in range(0, len(lmList) - 1):
                x1, y1 = lmList[i][1], lmList[i][2]
                x2, y2 = lmList[i + 1][1], lmList[i + 1][2]
                cv2.line(landmarks_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            lenFingScale = math.hypot(x2 - x1, y2 - y1)

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

            cv2.line(landmarks_frame, (x1, y1), (x2, y2),
                     (255 * (1 - fingScaleVal), 0, 255 * fingScaleVal), 2)
            cv2.circle(landmarks_frame, (x1, y1), 2, (0, 255, 0), cv2.FILLED)
            cv2.circle(landmarks_frame, (x2, y2), 2, (0, 255, 0), cv2.FILLED)

            if fingScaleVal < 0.15:
                cv2.circle(landmarks_frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            elif fingScaleVal > 0.85:
                cv2.circle(landmarks_frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
            else:
                cv2.circle(landmarks_frame, (cx, cy), 2, (255, 255, 255), cv2.FILLED)

        frames.append(landmarks_frame)

    cap.release()
    cv2.destroyAllWindows()

    normalize = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0014, 0.0014, 0.0015], std=[0.0117, 0.0119, 0.0123]),
    ])
    normalized_frames = torch.stack([normalize(frame) for frame in frames])
    return normalized_frames.permute(1, 0, 2, 3).unsqueeze(0)


def infer_gesture(model, video_path, device):
    """Run inference on a single video and return the predicted gesture label."""
    model.to(device)
    processed_video = preprocess_video(video_path).to(device)
    with torch.no_grad():
        outputs = model(processed_video)
    predicted_class_idx = torch.argmax(outputs, dim=1).item()
    return INT_TO_LABEL.get(predicted_class_idx, f"Unknown class {predicted_class_idx}")


if __name__ == "__main__":
    args = parse_args()
    model, device = load_model(args.model_path, num_classes=args.num_classes)
    predicted_gesture = infer_gesture(model, args.video_path, device)
    print("Predicted Gesture:", predicted_gesture)
