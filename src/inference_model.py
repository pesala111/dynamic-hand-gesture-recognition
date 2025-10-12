"""
This code is designed to evaluate the performance of the trained model in real-time scenarios.
"""
import cv2
import os
import numpy as np
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import HandTrackingModule as htm
import math


# ResNet3D model class
class ResNet3D(nn.Module):
    def __init__(self, num_classes):
        super(ResNet3D, self).__init__()

        # Loading the pre-trained ResNet3D model
        self.resnet3d = models.video.r3d_18(pretrained=True)

        num_features = self.resnet3d.fc.in_features

        # No dropout layer, just the Linear layer
        self.resnet3d.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet3d(x)

# Initialize hand detector
detector = htm.handDetector(detectionCon=1)


# Load trained model function
def load_model(model_path, num_classes=6):
    model = ResNet3D(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

fingScaleVal = 0
widthCap, heightCap = 640, 460
pTime = 0

# Preprocess video
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    widthCap, heightCap = 640, 460

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        landmarks_frame = np.zeros_like(img)

        # HandTrackingModule processing
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            # Provided code starts here
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
            # Provided code ends here

            frames.append(landmarks_frame)
        cv2.imshow("landmarks_frame", landmarks_frame)


    cap.release()
    cv2.destroyAllWindows()

    # Normalization
    normalize = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0014, 0.0014, 0.0015], std=[0.0117, 0.0119, 0.0123])
    ])
    normalized_frames = torch.stack([normalize(frame) for frame in frames])
    return normalized_frames.permute(1, 0, 2, 3).unsqueeze(0)

# Inference function
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def infer_gesture(model, video_path, device = device):
    model.to(device)
    processed_video = preprocess_video(video_path).to(device)
    with torch.no_grad():
        outputs = model(processed_video)
    predicted_class_idx = torch.argmax(outputs, dim=1).item()
    int_to_label = {v: k for k, v in label_to_int.items()}
    return int_to_label[predicted_class_idx]

# labels of the trained gesture classes in order
# modify according to it
label_to_int = {'10_Palm_Opening': 0,
                '11_Palm_Shake': 1,
                '09_Pulling': 2,
                '08_Pointing': 3,
                '16_CCW_Rotation': 4}
# load trained model
trained_model_path = 'input path to trained model(gesture_model_V3.pth)'
# load sample gesture video
video_path = "input any sample gesture video"
model = load_model(trained_model_path)
predicted_gesture = infer_gesture(model, video_path)
print("Predicted Gesture:", predicted_gesture)
