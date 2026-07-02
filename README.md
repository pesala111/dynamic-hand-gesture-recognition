# Dynamic Hand Gesture Recognition

Real-time dynamic hand gesture recognition using deep learning. The model is built on a pretrained **ResNet-3D (r3d_18)** architecture with PyTorch, and uses **MediaPipe** for hand landmark extraction. The system classifies gestures across **17 classes** from video data.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [License](#license)

---

## Overview

This project implements an end-to-end pipeline for dynamic hand gesture recognition:

1. **Hand Landmark Extraction** — Raw gesture videos are processed using MediaPipe to extract hand landmarks, which are rendered on a blank background to reduce noise.
2. **Frame Resampling** — Videos are resampled to a uniform frame length to ensure consistency during training.
3. **Model Training** — A ResNet-3D model is fine-tuned on the processed dataset with data augmentation (rotation and translation).
4. **Inference** — A real-time inference module loads the trained model and classifies live or recorded gestures.

---

## Project Structure

```
dynamic-hand-gesture-recognition/
├── data/
│   └── hand_gesture_dataset/       # 17 gesture class folders, each containing .avi video clips
├── docs/
│   ├── gesture-detection-literature-review.pdf
│   ├── project-documentation.pdf
│   └── system-specifications.xlsx
├── src/
│   ├── hand_tracking.py            # HandDetector module using MediaPipe
│   ├── landmark_extraction.py      # Extracts hand landmarks from raw gesture videos
│   ├── frame_extraction.py         # Resamples videos to a uniform frame length
│   ├── recognition_model.py        # Model definition, training, and evaluation
│   └── inference_model.py          # Real-time inference on new gesture videos
├── .gitignore
├── CHANGELOG.md
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Dataset

The dataset consists of **17 gesture classes**, each containing multiple video clips. Videos capture hand landmarks extracted via MediaPipe and rendered against a black background for uniformity.

| # | Gesture Class |
|---|--------------|
| 01 | Horizontal Swiping |
| 02 | Swiping Up |
| 03 | Swiping Down |
| 04 | V-Swiping Left |
| 05 | V-Swiping Right |
| 06 | V-Swiping Up |
| 07 | V-Swiping Down |
| 08 | Pointing |
| 09 | Pulling |
| 10 | Palm Opening |
| 11 | Palm Shake |
| 12 | Peace Sign |
| 13 | Three Finger Open |
| 14 | Pushing |
| 15 | Clockwise Rotation |
| 16 | Counter-Clockwise Rotation |
| 17 | Five Finger Closure |

All videos share a **uniform frame length** to ensure consistency during model training and evaluation.

---

## Installation

### Prerequisites

- Python 3.8+
- A CUDA-compatible GPU (recommended for training)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/pesala111/dynamic-hand-gesture-recognition.git
cd dynamic-hand-gesture-recognition
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 - Extract Hand Landmarks

Run landmark_extraction.py to extract hand landmarks from raw gesture videos:

```bash
python src/landmark_extraction.py
```

Update input_directory and output_directory inside the script to point to your dataset paths.

### Step 2 - Resample Video Frames

Run frame_extraction.py to normalize all videos to a uniform frame length:

```bash
python src/frame_extraction.py
```

### Step 3 - Train the Model

Run recognition_model.py to train the ResNet-3D model on the processed dataset:

```bash
python src/recognition_model.py
```

Update root_dir inside the script to point to your dataset root directory before running.

### Step 4 - Run Inference

Run inference_model.py to classify gestures from a new video:

```bash
python src/inference_model.py
```

---

## Documentation

Supporting documents are available in the docs/ folder:

- gesture-detection-literature-review.pdf - Literature review covering related work in gesture recognition
- project-documentation.pdf - Full project documentation
- system-specifications.xlsx - System specifications and requirements

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
