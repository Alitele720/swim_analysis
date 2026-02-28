import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "swimmer_pose_best.pt")
# === 添加路径 ===
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from preprocess import preprocess_frame


def process_swimming_rate(video_path, output_path='output_rate.mp4'):
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps,
                          (w, h))

    wrist_y_history = []
    frame_indices = []

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # === 使用统一预处理 ===
        frame_processed = preprocess_frame(frame)

        results = model(frame_processed, verbose=False)
        annotated_frame = results[0].plot()

        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            keypoints = results[0].keypoints.data[0].cpu().numpy()

            right_wrist_y = keypoints[10][1]
            conf = keypoints[10][2]

            if conf > 0.5:
                wrist_y_history.append(right_wrist_y)
                frame_indices.append(frame_count)

        out.write(annotated_frame)
        cv2.imshow('Stroke Analysis', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("视频分析完成")


if __name__ == "__main__":
    process_swimming_rate("test.mp4")