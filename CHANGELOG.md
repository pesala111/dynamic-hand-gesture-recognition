# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Planned
- Add unit tests
- Add demo GIF to README
- Publish model performance benchmarks

---

## [1.1.0] - 2026-07-03

### Added
- `argparse` CLI support in `recognition_model.py`, `landmark_extraction.py`, `frame_extraction.py`, and `inference_model.py`
- MIT License and CHANGELOG
- Modular function structure in `recognition_model.py` (`load_data`, `build_dataloaders`, `train`, `evaluate`, `plot_training_curves`, `plot_confusion_matrix`)
- `if __name__ == "__main__"` guards across all scripts

### Fixed
- `number_of_classes` off-by-one bug in `recognition_model.py`
- Hardcoded Colab and local paths replaced with `argparse` arguments
- `pretrained=True` deprecation replaced with `weights=R3D_18_Weights.DEFAULT`
- Incomplete `label_to_int` in `inference_model.py` now covers all 17 gesture classes
- Architecture mismatch between training and inference `ResNet3D` (Dropout head now consistent)
- `class handDetector` renamed to `class HandDetector` (PEP 8)
- MediaPipe `Hands()` positional args replaced with keyword args
- Live debug `print(lmList[4])` removed from `hand_tracking.py`
- `seaborn` added to `requirements.txt`; unused `pandas` and `tqdm` removed

### Changed
- README Usage section updated with correct CLI syntax for all four pipeline scripts

---

## [1.0.0] - 2025-10-12

### Added
- MIT License
- `.gitignore` for Python and project housekeeping
- `requirements.txt` with all project dependencies

### Changed
- Reorganized project structure: source files moved to `src/`, dataset to `data/`, documentation to `docs/`
- Renamed source files for consistency (`Landmarks_extraction.py` → `landmark_extraction.py`, `Frame_resample.py` → `frame_extraction.py`)

---

## [0.1.0] - 2023-09-18

### Added
- Initial implementation of dynamic hand gesture recognition
- ResNet-3D (r3d_18) model with MediaPipe hand landmark extraction
- 17-class gesture dataset support
- Data augmentation: rotation and translation transforms
- Early stopping and ReduceLROnPlateau scheduler
