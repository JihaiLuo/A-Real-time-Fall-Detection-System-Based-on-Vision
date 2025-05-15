# Fall Detection System

This project implements a real-time fall detection system using pose estimation (MediaPipe) and a Convolutional Neural Network (CNN).

Dataset: https://fenix.ur.edu.pl/~mkepski/ds/uf.html (RGB data Camera 0)

Demonstration video available at this link: 

https://youtu.be/FJpPpzc6TMc

https://youtu.be/QPlWs5TkxLM

The first video shows fall detection of a single person in an environment with occlusions.
The second video demonstrates detection when two people are present and one of them falls.

---

##  Project Structure

```
.
├── cnn_model.py                   # CNN model definition
├── extract_pose_from_frames.py    # Script to extract pose sequences from URFD frames
├── train_model.py                 # Script to train the CNN model
├── fall_cnn_model.pth             # Saved trained model
├── classification_report_eval.py  # Script to generate confusion matrix and evaluation metrics
├── media_pipe_visualization.py    # (Optional) Visualize pose landmarks on image
├── realtime_detection.py          # Real-time fall detection using webcam
├── requirements.txt               # List of required packages
└── README.md                      # This instruction file
```

---

## Installation

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Extract Pose Data

```bash
python extract_pose_from_frames.py
```

Converts video frames to pose sequences saved in `.npy` format under `pose_data/`.

---

### 2. Train CNN Model

```bash
python train_model.py
```

Trains a classifier and saves the model to `fall_cnn_model.pth`.

---

### 3. Evaluate Model

```bash
python classification_report_eval.py
```

Prints precision, recall, F1-score, and confusion matrix.

---

### 4. Real-Time Fall Detection

```bash
python realtime_detection.py
```

Launches webcam input and performs live fall classification.

---

## Requirements

See `requirements.txt`:

- opencv-python
- mediapipe
- numpy
- torch
- matplotlib
- scikit-learn

---

## Notes

- Ensure webcam access is allowed and functional.
- Input videos/images should clearly show the full human body for optimal pose extraction.
