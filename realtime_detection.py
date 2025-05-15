import cv2
import torch
import numpy as np
import mediapipe as mp
from cnn_model import FallDetectionCNN
from collections import deque

SEQUENCE_LENGTH = 30
device = torch.device("cpu")

model = FallDetectionCNN()
model.load_state_dict(torch.load("fall_cnn_model.pth", map_location=device))
model.to(device)
model.eval()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

pose_buffer = deque(maxlen=SEQUENCE_LENGTH)

cap = cv2.VideoCapture(0)
fall_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    keypoints = []
    if result.pose_landmarks:
        for lm in result.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y])
    else:
        keypoints = [0.0] * 66

    pose_buffer.append(keypoints)

    if len(pose_buffer) == SEQUENCE_LENGTH:
        input_tensor = torch.tensor([list(pose_buffer)], dtype=torch.float32).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            fall_detected = (pred == 1)

    label = "FALL" if fall_detected else "NON-FALL"
    color = (0, 0, 255) if fall_detected else (0, 255, 0)
    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    cv2.imshow("Real-time Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
