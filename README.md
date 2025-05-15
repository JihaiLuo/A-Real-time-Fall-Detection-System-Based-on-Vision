Fall Detection System

This project implements a real-time fall detection system using pose estimation (MediaPipe) and a Convolutional Neural Network (CNN).

Dataset: https://fenix.ur.edu.pl/~mkepski/ds/uf.html (RGB data)

---
 Project Structure
├── URFD_frames/                # Raw dataset: frame-by-frame images from UR Fall Detection dataset
├── pose_data/                  # Processed pose data (output from MediaPipe)
├── cnn_model.py                # CNN model definition
├── extract_pose_from_frames.py # Script to extract pose sequences from URFD frames
├── train_model.py              # Script to train the CNN model
├── fall_cnn_model.pth          # Saved trained model
├── classification_report_eval.py # Script to generate confusion matrix and evaluation metrics
├── media_pipe_visualization.py # (Optional) Visualize pose landmarks on image
├── realtime_detection.py       # Real-time fall detection using webcam
├── requirements.txt            # List of required packages
└── README.md                   # This instruction file
---
Installation
Create a virtual environment and install dependencies:
```bash
pip install -r requirements.txt
```
---
How to Run
1. Extract Pose Data
Run this to extract pose landmarks from URFD dataset and save as `.npy` files:

```bash
python extract_pose_from_frames.py
```

This will generate pose sequences under `pose_data/` directory.
---
2. Train CNN Model
Train the fall detection model on extracted pose sequences:

```bash
python train_model.py
```

The model will be saved as `fall_cnn_model.pth`.
---

3. Evaluate Model (Optional)
Evaluate model accuracy, precision, recall, and confusion matrix:

```bash
python classification_report_eval.py
```
---
4. Real-Time Detection

Use webcam to detect falls in real time:

```bash
python realtime_detection.py
```

If someone falls in front of the camera, the system displays a red "FALL" alert; otherwise, it shows a green "NON-FALL".

---
 Requirements

Make sure the following are installed (see `requirements.txt`):

- Python 3.9+
- opencv-python
- mediapipe
- torch
- numpy
- matplotlib
- scikit-learn

---