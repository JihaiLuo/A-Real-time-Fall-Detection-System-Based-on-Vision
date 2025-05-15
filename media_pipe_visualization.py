import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def choose_image_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.png *.jpeg")]
    )
    return file_path

def visualize_pose_from_image():
    img_path = choose_image_file()
    if not img_path:
        print("❌ No image selected.")
        return

    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ Failed to read image from: {img_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            print("❌ No pose landmarks detected.")
            return

        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        cv2.imshow("Pose Visualization", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_pose_from_image()
