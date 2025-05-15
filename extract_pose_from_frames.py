import os
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def extract_sequence_from_folder(folder_path, sequence_length=30):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))])
    keypoint_list = []

    for fname in files:
        img_path = os.path.join(folder_path, fname)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)

        keypoints = []
        if result.pose_landmarks:
            for lm in result.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y])
        else:
            keypoints = [0.0] * 66  

        keypoint_list.append(keypoints)

    return np.array(keypoint_list)

def process_all_folders(base_dir, output_root="pose_data", sequence_length=30):
    os.makedirs(output_root, exist_ok=True)
    folder_list = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    for folder in folder_list:
        label = "fall" if folder.lower().startswith("fall") else "nonfall"
        input_path = os.path.join(base_dir, folder)
        output_path = os.path.join(output_root, label)
        os.makedirs(output_path, exist_ok=True)

        print(f"🔍 Processing {folder} → label: {label}")
        data = extract_sequence_from_folder(input_path, sequence_length)
        num_samples = 0

        for start in range(0, len(data) - sequence_length + 1, sequence_length):
            seq = data[start:start + sequence_length]
            save_path = os.path.join(output_path, f"{folder}_seq{num_samples}.npy")
            np.save(save_path, seq)
            num_samples += 1

        print(f"Saved {num_samples} samples to {output_path}\n")

if __name__ == "__main__":
    base_dir = "URFD_frames"
    process_all_folders(base_dir)
