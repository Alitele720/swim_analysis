import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os


class SwimCoachAI:
    def __init__(self, model_type='n'):
        """
        初始化 AI 教练
        :param model_type: 'n', 's', 'm', 'l', 'x' (推荐 x 获取最高精度)
        """
        print(f"正在初始化 AI 系统 (加载 YOLOv8{model_type}-pose 模型)...")
        self.model = YOLO(f'yolov8{model_type}-pose.pt')
        print("模型加载完毕！")

    def _calculate_angle(self, a, b, c):
        """辅助函数：计算三点夹角"""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0: angle = 360 - angle
        return angle

    # def _enhance_frame(self, frame):
    #     """辅助函数：图像增强 (对比度/亮度)"""
    #     return cv2.convertScaleAbs(frame, alpha=1.5, beta=10)

    def _enhance_frame(self, frame):
        """
        辅助函数：图像增强 (V2.0 CLAHE版)
        专门解决：水花和人体颜色相近、画面模糊导致的识别失败
        """
        # 1. 转换到 LAB 色彩空间
        # (LAB 将亮度(L)和颜色(AB)分离，我们在不破坏颜色的情况下只增强亮度细节)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 2. 应用 CLAHE (自适应直方图均衡化)
        # clipLimit: 对比度限制 (设为 2.0-4.0，太高会引入噪点，推荐 3.0)
        # tileGridSize: 网格大小 (8x8 适合大部分视频)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # 3. 合并通道并转回 BGR
        limg = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # (可选) 4. 稍微锐化一点边缘 (应对模糊)
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        # enhanced_img = cv2.filter2D(enhanced_img, -1, kernel)

        return enhanced_img

    def _repair_data_series(self, raw_data, valid_indices, total_frames):
        """辅助函数：数据插值与修复"""
        if len(raw_data) < 5:
            return np.array([])

        valid_frames = np.array(valid_indices)
        valid_values = np.array(raw_data)

        # 线性插值
        full_frames = np.arange(valid_frames[0], valid_frames[-1] + 1)
        full_values_interp = np.interp(full_frames, valid_frames, valid_values)

        # 平滑处理
        window_size = 15 if len(full_values_interp) > 20 else 5
        window = np.hanning(window_size)
        window = window / window.sum()
        smoothed_values = np.convolve(full_values_interp, window, mode='same')

        return full_frames, smoothed_values

    # ======================================================
    # 功能模块 1: 实时姿态视觉反馈 (Visual Feedback)
    # ======================================================
    def analyze_pose(self, video_path, output_path='output_visual.mp4', enable_enhance=False):
        mode_str = "开启" if enable_enhance else "关闭"
        print(f"\n[模式 1] 启动视觉分析: {video_path} (图像增强: {mode_str})...")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return

        w, h = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        print("按 'q' 键可退出实时预览")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # 根据用户选择决定是否增强
            if enable_enhance:
                frame_input = self._enhance_frame(frame)
            else:
                frame_input = frame

            results = self.model(frame_input, verbose=False)
            annotated_frame = results[0].plot()

            if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                kp = results[0].keypoints.data[0].cpu().numpy()
                sh, el, wr = kp[6][:2], kp[8][:2], kp[10][:2]  # 右臂
                conf_check = kp[6][2] * kp[8][2] * kp[10][2]

                if conf_check > 0.5:
                    angle = self._calculate_angle(sh, el, wr)
                    cv2.putText(annotated_frame, f"{int(angle)} deg",
                                (int(el[0]), int(el[1]) - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    if angle > 160:
                        status, color = "WARN: Straight Arm", (0, 0, 255)
                    else:
                        status, color = "Good: High Elbow", (0, 255, 0)

                    cv2.putText(annotated_frame, status, (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            out.write(annotated_frame)
            cv2.imshow('Visual Feedback', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"视觉分析完成，视频已保存至: {output_path}")

    # ======================================================
    # 功能模块 2: 划频分析 (Stroke Rate Analysis)
    # ======================================================
    def analyze_rate(self, video_path, enable_enhance=False):
        mode_str = "开启" if enable_enhance else "关闭"
        print(f"\n[模式 2] 进行划频分析: {video_path} (图像增强: {mode_str})...")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        wrist_history = []
        frame_indices = []
        frame_count = 0

        print("按 'q' 键可提前结束分析")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1

            # 根据用户选择决定是否增强
            if enable_enhance:
                frame_input = self._enhance_frame(frame)
            else:
                frame_input = frame

            results = self.model(frame_input, verbose=False)

            # 实时显示骨架
            annotated_frame = results[0].plot()

            if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                kp = results[0].keypoints.data[0].cpu().numpy()
                wrist_y = kp[10][1]  # 右腕 Y

                if kp[10][2] > 0.5:
                    wrist_history.append(wrist_y)
                    frame_indices.append(frame_count)
                    cv2.circle(annotated_frame, (int(kp[10][0]), int(wrist_y)), 10, (0, 0, 255), -1)

            cv2.putText(annotated_frame, f"Analyzing Rate: Frame {frame_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow('Stroke Rate Analysis (Processing)', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

        valid_y = -1 * np.array(wrist_history)
        full_frames, y_smoothed = self._repair_data_series(valid_y, frame_indices, frame_count)

        if len(full_frames) == 0:
            print("错误：采集到的有效数据不足。")
            return

        data_range = np.max(y_smoothed) - np.min(y_smoothed)
        threshold = np.min(y_smoothed) + (data_range * 0.4)
        peaks, _ = find_peaks(y_smoothed, height=threshold, distance=int(fps * 0.6))

        num_strokes = len(peaks)
        duration = (full_frames[-1] - full_frames[0]) / fps
        spm = (num_strokes / duration) * 60 if duration > 0 else 0

        print(f"\n分析结果: 检测到 {num_strokes} 次划水，平均划频 (SPM): {spm:.1f}")

        plt.figure(figsize=(12, 6))
        plt.plot(full_frames, y_smoothed, label='Smoothed Curve', color='#1f77b4', linewidth=2)
        plt.plot(full_frames[peaks], y_smoothed[peaks], "x", color='red', markersize=12, label='Stroke Peak')
        plt.axhline(y=threshold, color='green', linestyle='--', label='Threshold')
        plt.title(f'Stroke Rate Analysis (SPM: {spm:.1f})')
        plt.xlabel('Frame Index')
        plt.ylabel('Wrist Vertical Position')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    # ======================================================
    # 功能模块 3: 专业动作对比 (Pro Comparison / DTW)
    # ======================================================
    def _extract_features(self, video_path, window_title="Feature Extraction", enable_enhance=False):
        """功能3的内部辅助函数：提取角度序列"""
        cap = cv2.VideoCapture(video_path)
        angles = []
        indices = []
        frame_count = 0

        print(f"正在提取特征: {os.path.basename(video_path)}...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            if frame_count % 2 != 0: continue

            # 根据用户选择决定是否增强
            if enable_enhance:
                frame_input = self._enhance_frame(frame)
            else:
                frame_input = frame

            results = self.model(frame_input, verbose=False)

            annotated_frame = results[0].plot()
            cv2.putText(annotated_frame, f"Extracting: {os.path.basename(video_path)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                kp = results[0].keypoints.data[0].cpu().numpy()
                sh, el, wr = kp[6][:2], kp[8][:2], kp[10][:2]
                if kp[6][2] * kp[8][2] * kp[10][2] > 0.5:
                    angles.append(self._calculate_angle(sh, el, wr))
                    indices.append(frame_count)

            cv2.imshow(window_title, annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyWindow(window_title)

        full_frames, smooth_angles = self._repair_data_series(angles, indices, frame_count)
        return smooth_angles.reshape(-1, 1) if len(smooth_angles) > 0 else None

    def compare_form(self, user_video, pro_video, enable_enhance=False):
        mode_str = "开启" if enable_enhance else "关闭"
        print(f"\n[模式 3] 开始对比 (图像增强: {mode_str})...")

        # 注意：这里我们将增强选项同时应用到两个视频
        # 如果你的专业视频很清晰而用户视频很模糊，这里可能需要更细的逻辑
        # 但通常对齐时保持处理方式一致是比较科学的
        seq_user = self._extract_features(user_video, "Processing User Video", enable_enhance)
        seq_pro = self._extract_features(pro_video, "Processing Pro Video", enable_enhance)

        if seq_user is None or seq_pro is None:
            print("错误：特征提取失败（可能视频中未检测到人）。")
            return

        print("正在计算 DTW 相似度...")
        distance, path = fastdtw(seq_user, seq_pro, dist=euclidean)
        avg_error = distance / len(path)
        score = max(0, 100 - avg_error * 1.5)

        print(f"AI 评分: {score:.1f} / 100 (平均帧误差: {avg_error:.2f} 度)")

        plt.figure(figsize=(10, 6))
        user_warped = [seq_user[idx_u][0] for idx_u, idx_p in path]
        pro_warped = [seq_pro[idx_p][0] for idx_u, idx_p in path]

        plt.plot(user_warped, label='User (Aligned)', color='#1f77b4')
        plt.plot(pro_warped, label='Pro (Aligned)', color='orange', linestyle='--')
        plt.fill_between(range(len(user_warped)), user_warped, pro_warped, color='red', alpha=0.2, label='Error')
        plt.title(f'Form Comparison (Score: {score:.1f})')
        plt.legend()
        plt.show()


# ======================================================
# 主入口菜单 (汉化 + 增强选项版)
# ======================================================
if __name__ == "__main__":
    coach = SwimCoachAI(model_type='x')

    default_video = "test.mp4"
    default_pro = "pro.mp4"

    while True:
        print("\n========= AI 游泳教练系统 =========")
        print("1. 视觉动作反馈 (高肘检测/实时画面)")
        print("2. 划频节奏分析 (SPM折线图/数据修复)")
        print("3. 专业动作对比 (DTW算法评分)")
        print("q. 退出系统")
        print("===================================")

        choice = input("请选择功能 (输入 1/2/3/q): ").lower()

        if choice == 'q':
            print("系统已退出。")
            break

        # 1. 获取视频路径
        target_video = input(f"请输入视频路径 [默认: {default_video}]: ").strip()
        if not target_video: target_video = default_video

        if not os.path.exists(target_video):
            print(f"错误：找不到文件 '{target_video}'，请检查路径。")
            continue

        # 2. 询问是否开启图像增强 (仅在需要时询问)
        use_enhance = input("是否启用图像增强 (针对模糊/水下视频)? (y/n) [默认: n]: ").strip().lower() == 'y'

        if choice == '1':
            coach.analyze_pose(target_video, enable_enhance=use_enhance)
        elif choice == '2':
            coach.analyze_rate(target_video, enable_enhance=use_enhance)
        elif choice == '3':
            pro_video = input(f"请输入标准(专业)视频路径 [默认: {default_pro}]: ").strip()
            if not pro_video: pro_video = default_pro

            if os.path.exists(pro_video):
                coach.compare_form(target_video, pro_video, enable_enhance=use_enhance)
            else:
                print(f"错误：找不到标准视频文件 '{pro_video}'。")
        else:
            print("无效的选择，请重新输入。")