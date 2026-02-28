# core_engine.py
import cv2
import numpy as np
from ultralytics import YOLO


class PoseEngine:
    def __init__(self, model_type='x'):
        print(f"正在加载 YOLOv8{model_type}-pose 模型...")
        self.model = YOLO(f'yolov8{model_type}-pose.pt')

    @staticmethod
    def calculate_angle(a, b, c):
        """计算三点夹角"""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle

    @staticmethod
    def enhance_frame(frame):
        """CLAHE 图像增强 (解决水下和水花模糊问题)"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def process_video(self, video_path, enable_enhance=False, frame_skip=1):
        """
        核心视频生成器：负责读取、增强、推理，并动态输出最优的手臂数据
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频 {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 第一次 yield 返回视频基础信息
        yield cap, fps, w, h

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1

            if frame_count % frame_skip != 0:
                continue

            # 1. 图像增强
            input_frame = self.enhance_frame(frame) if enable_enhance else frame

            # 2. YOLO 推理
            results = self.model(input_frame, verbose=False)
            annotated_frame = results[0].plot()

            # 3. 动态左右臂识别逻辑
            best_arm = None
            if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                kp = results[0].keypoints.data[0].cpu().numpy()

                # 计算左右臂置信度 (肩 * 肘 * 腕)
                left_conf = kp[5][2] * kp[7][2] * kp[9][2]  # 左侧: 5, 7, 9
                right_conf = kp[6][2] * kp[8][2] * kp[10][2]  # 右侧: 6, 8, 10

                if max(left_conf, right_conf) > 0.5:
                    if right_conf >= left_conf:
                        best_arm = {'side': 'Right', 'sh': kp[6][:2], 'el': kp[8][:2], 'wr': kp[10][:2],
                                    'wr_y': kp[10][1]}
                    else:
                        best_arm = {'side': 'Left', 'sh': kp[5][:2], 'el': kp[7][:2], 'wr': kp[9][:2], 'wr_y': kp[9][1]}

            # 每次循环产出当前帧的数据
            yield frame_count, annotated_frame, best_arm

        cap.release()