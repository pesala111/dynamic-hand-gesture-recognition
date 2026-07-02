# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Planned
- Add command-line argument support via `argparse`
- - Add unit tests
  - - Add demo GIF to README
    - - Publish model performance benchmarks
     
      - ---

      ## [1.0.0] - 2025-10-12

      ### Added
      - MIT License
      - - `.gitignore` for Python and project housekeeping
        - - `requirements.txt` with all project dependencies
         
          - ### Changed
          - - Reorganized project structure: source files moved to `src/`, dataset to `data/`, documentation to `docs/`
            - - Renamed source files for consistency (`Landmarks_extraction.py` -> `landmark_extraction.py`, `Frame_resample.py` -> `frame_extraction.py`)
             
              - ---

              ## [0.1.0] - 2023-09-18

              ### Added
              - Initial implementation of dynamic hand gesture recognition model
              - - ResNet-3D (r3d_18) based classification model using PyTorch
                - - Hand landmark extraction pipeline using MediaPipe
                  - - Frame extraction and resampling scripts
                    - - Inference module for real-time gesture recognition
                      - - Dataset with 17 gesture classes
                        - - Project documentation and system specifications in `docs/`
                          - 
