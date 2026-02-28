import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "swimmer_pose_best.pt")
# === 添加路径 ===
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from preprocess import preprocess_frame


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def extract_pose_sequence(video_path, description="Video"):
    print(f"Extracting features from: {description} ...")

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(video_path)

    raw_angles = []
    frame_indices = []

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 2 != 0:
            continue

        # === 使用统一预处理 ===
        frame_processed = preprocess_frame(frame)

        results = model(frame_processed, verbose=False)

        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            kp = results[0].keypoints.data[0].cpu().numpy()

            shoulder = kp[6][:2]
            elbow = kp[8][:2]
            wrist = kp[10][:2]

            conf_check = kp[6][2] * kp[8][2] * kp[10][2]

            if conf_check > 0.5:
                angle = calculate_angle(shoulder, elbow, wrist)
                raw_angles.append(angle)
                frame_indices.append(frame_count)

    cap.release()

    if len(raw_angles) < 10:
        return np.array([])

    valid_frames = np.array(frame_indices)
    valid_angles = np.array(raw_angles)

    full_frames = np.arange(valid_frames[0], valid_frames[-1] + 1)
    full_angles_interp = np.interp(full_frames, valid_frames, valid_angles)

    window_size = 5
    window = np.ones(window_size) / window_size
    full_angles_smooth = np.convolve(full_angles_interp, window, mode='same')

    return full_angles_smooth.reshape(-1, 1)


def compare_swimming_form(user_video, pro_video):
    seq_user = extract_pose_sequence(user_video, "User Video")
    seq_pro = extract_pose_sequence(pro_video, "Pro Video")

    if len(seq_user) == 0 or len(seq_pro) == 0:
        print("Error: Failed to extract skeleton data.")
        return

    distance, path = fastdtw(seq_user, seq_pro, dist=euclidean)
    avg_error = distance / len(path)
    score = max(0, 100 - avg_error * 1.5)

    print(f"Score: {score:.1f}")


if __name__ == "__main__":
    compare_swimming_form("test.mp4", "pro.mp4")