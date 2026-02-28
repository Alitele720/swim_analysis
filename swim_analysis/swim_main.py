# swim_main.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from core_engine import PoseEngine  # 引入刚刚写的核心引擎


class SwimCoachAI:
    def __init__(self):
        self.engine = PoseEngine(model_type='x')

    def analyze_pose(self, video_path, output_path='out_visual.mp4', enable_enhance=False):
        """模块 1: 视觉动作反馈 (大幅简化)"""
        stream = self.engine.process_video(video_path, enable_enhance)
        cap, fps, w, h = next(stream)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        for frame_count, annotated_frame, best_arm in stream:
            if best_arm:
                # 使用动态选出的手臂计算角度
                angle = self.engine.calculate_angle(best_arm['sh'], best_arm['el'], best_arm['wr'])

                color, status = ((0, 0, 255), f"WARN: Straight Arm ({best_arm['side']})") if angle > 160 else (
                (0, 255, 0), f"Good: High Elbow ({best_arm['side']})")

                # 绘制信息
                el_pt = best_arm['el']
                cv2.putText(annotated_frame, f"{int(angle)} deg", (int(el_pt[0]), int(el_pt[1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(annotated_frame, status, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            out.write(annotated_frame)
            cv2.imshow('Visual Feedback', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        out.release()
        cv2.destroyAllWindows()

    def analyze_rate(self, video_path, enable_enhance=False):
        """模块 2: 划频分析 (使用动态手腕Y坐标)"""
        stream = self.engine.process_video(video_path, enable_enhance)
        cap, fps, _, _ = next(stream)

        wrist_history, frame_indices = [], []

        for frame_count, annotated_frame, best_arm in stream:
            if best_arm:
                # 记录有效的手腕Y坐标 (取反以符合直觉)
                wrist_history.append(-1 * best_arm['wr_y'])
                frame_indices.append(frame_count)

            cv2.imshow('Analyzing Stroke Rate...', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cv2.destroyAllWindows()

        # 数据平滑与寻峰逻辑 (保持你原有的优秀逻辑)
        if len(wrist_history) < 10:
            print("有效数据不足")
            return

        full_frames = np.arange(frame_indices[0], frame_indices[-1] + 1)
        y_interp = np.interp(full_frames, frame_indices, wrist_history)
        y_smoothed = np.convolve(y_interp, np.hanning(15) / np.hanning(15).sum(), mode='same')

        threshold = np.min(y_smoothed) + (np.max(y_smoothed) - np.min(y_smoothed)) * 0.4
        peaks, _ = find_peaks(y_smoothed, height=threshold, distance=int(fps * 0.6))

        duration = (full_frames[-1] - full_frames[0]) / fps
        spm = (len(peaks) / duration) * 60 if duration > 0 else 0
        print(f"划水次数: {len(peaks)}, SPM: {spm:.1f}")

    def _extract_angle_sequence(self, video_path, enable_enhance=False, window_title="Extracting"):
        """
        内部特征提取器：消费 stream 数据，提取连续的手臂角度序列。
        这就是消费流数据的完美示例！完全不用写 cap.read() 和模型推理逻辑。
        """
        # frame_skip=2 可以加速提取过程，每两帧抽样一次
        stream = self.engine.process_video(video_path, enable_enhance, frame_skip=2)

        # 1. 弹出生成器的第一次 yield (获取视频基础信息，这里我们不需要保存视频，所以略过)
        next(stream)

        angles = []
        frame_indices = []

        # 2. 疯狂消费数据流！
        for frame_count, annotated_frame, best_arm in stream:
            if best_arm:
                # 直接从最优手臂中获取坐标并计算角度
                angle = self.engine.calculate_angle(best_arm['sh'], best_arm['el'], best_arm['wr'])
                angles.append(angle)
                frame_indices.append(frame_count)

            # 实时显示提取进度
            cv2.putText(annotated_frame, f"Task: {window_title}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('Feature Extraction', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cv2.destroyAllWindows()

        # 3. 经典的 SciPy 数据修复与平滑 (填补断点)
        if len(angles) < 5:
            return None

        full_frames = np.arange(frame_indices[0], frame_indices[-1] + 1)
        y_interp = np.interp(full_frames, frame_indices, angles)
        # 用汉宁窗平滑，消除 YOLO 识别带来的微小抖动
        y_smoothed = np.convolve(y_interp, np.hanning(9) / np.hanning(9).sum(), mode='same')

        # 返回 (N, 1) 形状供 DTW 算法使用
        return y_smoothed.reshape(-1, 1)

    def compare_form(self, user_video, pro_video, enable_enhance=False):
        """模块 3: 专业动作对比 (基于 DTW 算法)"""
        print(f"\n--- 开始提取【用户视频】动作特征 ---")
        seq_user = self._extract_angle_sequence(user_video, enable_enhance, "User Video")

        print(f"\n--- 开始提取【专业视频】动作特征 ---")
        seq_pro = self._extract_angle_sequence(pro_video, enable_enhance, "Pro Video")

        if seq_user is None or seq_pro is None:
            print("错误：特征提取失败，可能是视频中无人或有效帧过少。")
            return

        print("特征提取完毕，正在运行 DTW 动态时间规整...")
        distance, path = fastdtw(seq_user, seq_pro, dist=euclidean)

        # 评分公式
        avg_error = distance / len(path)
        score = max(0, 100 - avg_error * 1.5)

        print(f"=============================")
        print(f" AI 动作相似度评分: {score:.1f} / 100")
        print(f" 平均每帧角度误差: {avg_error:.2f} 度")
        print(f"=============================")

        # ========== Matplotlib 可视化 ==========
        user_warped = [seq_user[idx_u][0] for idx_u, idx_p in path]
        pro_warped = [seq_pro[idx_p][0] for idx_u, idx_p in path]

        plt.figure(figsize=(12, 6))
        plt.plot(user_warped, label='User Form (Aligned)', color='#1f77b4', linewidth=2)
        plt.plot(pro_warped, label='Pro Form (Aligned)', color='orange', linestyle='--', linewidth=2)

        # 填充红色表示误差区域
        plt.fill_between(range(len(user_warped)), user_warped, pro_warped, color='red', alpha=0.15, label='Error Gap')

        plt.title(f'Swimming Form DTW Alignment (AI Score: {score:.1f}/100)')
        plt.xlabel('Aligned Time Steps')
        plt.ylabel('Elbow Angle (Degrees)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import os

    # 实例化重构后的教练系统
    coach = SwimCoachAI()

    default_video = "test.mp4"
    default_pro = "pro.mp4"

    while True:
        print("\n========= AI 游泳教练系统 (Pro重构版) =========")
        print("1. 视觉动作反馈 (高肘检测/实时画面)")
        print("2. 划频节奏分析 (SPM折线图/数据修复)")
        print("3. 专业动作对比 (DTW算法评分)")
        print("q. 退出系统")
        print("===============================================")

        choice = input("请选择功能 (输入 1/2/3/q): ").lower()

        if choice == 'q':
            print("系统已退出。")
            break

        if choice not in ['1', '2', '3']:
            print("无效的选择，请重新输入。")
            continue

        # 1. 获取目标视频路径
        target_video = input(f"请输入测试视频路径 [默认: {default_video}]: ").strip()
        if not target_video: target_video = default_video

        if not os.path.exists(target_video):
            print(f"错误：找不到文件 '{target_video}'，请检查路径。")
            continue

        # 2. 询问是否开启图像增强
        use_enhance = input("是否启用 CLAHE 图像增强 (针对水花遮挡/模糊)? (y/n) [默认: n]: ").strip().lower() == 'y'

        # 3. 路由到对应的功能模块
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