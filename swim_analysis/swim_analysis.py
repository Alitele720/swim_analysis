import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "swimmer_pose_best.pt")


# === 关键新增：添加根目录路径，用于导入 preprocess.py ===
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from preprocess import preprocess_frame


# ==========================================
# 计算关节角度
# ==========================================
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


# ==========================================
# 主程序
# ==========================================
def process_swimming_analysis(video_path, output_path='output_analysis.mp4'):
    print("正在加载 YOLOv8n-pose 模型...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps,
                          (w, h))

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

            idx_shoulder, idx_elbow, idx_wrist = 6, 8, 10

            shoulder = keypoints[idx_shoulder][:2]
            elbow = keypoints[idx_elbow][:2]
            wrist = keypoints[idx_wrist][:2]

            conf_s = keypoints[idx_shoulder][2]
            conf_e = keypoints[idx_elbow][2]
            conf_w = keypoints[idx_wrist][2]

            if conf_s > 0.5 and conf_e > 0.5 and conf_w > 0.5:
                elbow_angle = calculate_angle(shoulder, elbow, wrist)

                cv2.putText(annotated_frame,
                            f"{int(elbow_angle)} deg",
                            (int(elbow[0]), int(elbow[1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2)

        cv2.putText(annotated_frame,
                    f"Frame: {frame_count}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2)

        out.write(annotated_frame)
        cv2.imshow('Swimming Posture Analysis', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"分析完成，视频已保存至 {output_path}")


if __name__ == "__main__":
    process_swimming_analysis("test.mp4")