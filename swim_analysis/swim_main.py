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


if __name__ == "__main__":
    coach = SwimCoachAI()
    # 测试运行
    coach.analyze_pose("test.mp4", enable_enhance=True)