import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from core_engine import PoseEngine  # 引入核心引擎


class SwimCoachAI:
    def __init__(self):
        self.engine = PoseEngine(model_type='x')

    # ================= 业务模块 1: 实时反馈 =================
    def analyze_pose(self, video_path, output_path='out_visual.mp4', enable_enhance=False):
        stream = self.engine.process_video(video_path, enable_enhance)
        cap, fps, w, h = next(stream)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        print("\n▶ 正在分析姿态... (在视频窗口按 'Q' 键可随时中止并返回菜单)")

        for frame_count, annotated_frame, best_arm in stream:
            if best_arm:
                angle = self.engine.calculate_angle(best_arm['sh'], best_arm['el'], best_arm['wr'])
                color, status = ((0, 0, 255), f"WARN: Straight Arm ({best_arm['side']})") if angle > 160 else (
                (0, 255, 0), f"Good: High Elbow ({best_arm['side']})")

                el_pt = best_arm['el']
                cv2.putText(annotated_frame, f"{int(angle)} deg", (int(el_pt[0]), int(el_pt[1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(annotated_frame, status, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            out.write(annotated_frame)
            cv2.imshow("Visual Feedback (Press 'Q' to abort)", annotated_frame)

            # 侦测 Q 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[提示] 🛑 用户手动中止了分析，正在返回主菜单...")
                break

        out.release()
        cv2.destroyAllWindows()

    # ================= 业务模块 2: 划频分析 =================
    def analyze_rate(self, video_path, enable_enhance=False):
        stream = self.engine.process_video(video_path, enable_enhance)
        cap, fps, _, _ = next(stream)

        wrist_history, frame_indices = [], []
        abort = False  # 中止标志

        print("\n▶ 正在提取划频数据... (在视频窗口按 'Q' 键可随时中止)")

        for frame_count, annotated_frame, best_arm in stream:
            if best_arm:
                wrist_history.append(-1 * best_arm['wr_y'])
                frame_indices.append(frame_count)

            cv2.imshow("Analyzing Stroke Rate (Press 'Q' to abort)", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[提示] 🛑 用户手动中止提取，已取消报告生成，返回主菜单...")
                abort = True
                break

        cv2.destroyAllWindows()

        if abort:
            return

        # 解决假性重复播放：明确打印错误原因
        if len(wrist_history) < 10:
            print("\n[错误] ❌ 有效手腕数据不足（模型未能清晰捕捉到手臂连续动作）。")
            print("建议：1. 开启图像增强 (选 y)\n      2. 尝试更清晰的视频片段\n已退回主菜单。")
            return

        print("\n⚙️ 数据提取完毕，正在生成划频报告图表...")

        full_frames = np.arange(frame_indices[0], frame_indices[-1] + 1)
        y_interp = np.interp(full_frames, frame_indices, wrist_history)

        # 动态自适应平滑窗口，防止短视频抛出异常
        window_size = 15 if len(y_interp) > 15 else (3 if len(y_interp) > 3 else 1)
        if window_size > 1:
            y_smoothed = np.convolve(y_interp, np.hanning(window_size) / np.hanning(window_size).sum(), mode='same')
        else:
            y_smoothed = y_interp

        threshold = np.min(y_smoothed) + (np.max(y_smoothed) - np.min(y_smoothed)) * 0.4
        peaks, _ = find_peaks(y_smoothed, height=threshold, distance=int(fps * 0.6))

        duration = (full_frames[-1] - full_frames[0]) / fps
        spm = (len(peaks) / duration) * 60 if duration > 0 else 0

        print(f"\n✅ 报告生成成功！\n检测到划水次数: {len(peaks)} 次\n平均划频: {spm:.1f} SPM")
        print("💡 提示：请关闭弹出的图表窗口，即可自动返回主菜单。")

        plt.figure(figsize=(10, 5))
        plt.plot(full_frames, y_smoothed, label='Smoothed Wrist Track', color='#1f77b4')
        plt.plot(full_frames[peaks], y_smoothed[peaks], "x", color='red', markersize=10, label='Stroke Peak')
        plt.axhline(y=threshold, color='green', linestyle='--', label='Threshold')
        plt.title(f'Stroke Rate Analysis (SPM: {spm:.1f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()  # 阻塞运行，关闭图表后才会跳出该函数返回菜单

    # ================= 业务模块 3: 动作对比 (DTW) =================
    def _extract_angle_sequence(self, video_path, enable_enhance=False, window_title="Extracting"):
        stream = self.engine.process_video(video_path, enable_enhance, frame_skip=2)
        next(stream)

        angles, frame_indices = [], []
        abort = False

        for frame_count, annotated_frame, best_arm in stream:
            if best_arm:
                angle = self.engine.calculate_angle(best_arm['sh'], best_arm['el'], best_arm['wr'])
                angles.append(angle)
                frame_indices.append(frame_count)

            cv2.putText(annotated_frame, f"Task: {window_title}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(f"{window_title} (Press 'Q' to abort)", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"\n[提示] 🛑 用户手动中止了 {window_title} 的提取。")
                abort = True
                break

        cv2.destroyAllWindows()

        if abort or len(angles) < 5:
            return None

        full_frames = np.arange(frame_indices[0], frame_indices[-1] + 1)
        y_interp = np.interp(full_frames, frame_indices, angles)

        window_size = 9 if len(y_interp) > 9 else 3
        y_smoothed = np.convolve(y_interp, np.hanning(window_size) / np.hanning(window_size).sum(), mode='same')
        return y_smoothed.reshape(-1, 1)

    def compare_form(self, user_video, pro_video, enable_enhance=False):
        print(f"\n--- 开始提取【用户视频】动作特征 ---")
        seq_user = self._extract_angle_sequence(user_video, enable_enhance, "User Video")
        if seq_user is None:
            print("\n[错误] ❌ 用户视频特征提取中止或失败，返回主菜单。")
            return

        print(f"\n--- 开始提取【专业视频】动作特征 ---")
        seq_pro = self._extract_angle_sequence(pro_video, enable_enhance, "Pro Video")
        if seq_pro is None:
            print("\n[错误] ❌ 专业视频特征提取中止或失败，返回主菜单。")
            return

        print("\n⚙️ 特征提取完毕，正在运行 DTW 动态时间规整...")
        distance, path = fastdtw(seq_user, seq_pro, dist=euclidean)

        avg_error = distance / len(path)
        score = max(0, 100 - avg_error * 1.5)

        print(f"=============================")
        print(f" AI 动作相似度评分: {score:.1f} / 100")
        print(f" 平均每帧角度误差: {avg_error:.2f} 度")
        print(f"=============================")
        print("💡 提示：请关闭弹出的图表窗口，即可自动返回主菜单。")

        user_warped = [seq_user[idx_u][0] for idx_u, idx_p in path]
        pro_warped = [seq_pro[idx_p][0] for idx_u, idx_p in path]

        plt.figure(figsize=(12, 6))
        plt.plot(user_warped, label='User Form (Aligned)', color='#1f77b4', linewidth=2)
        plt.plot(pro_warped, label='Pro Form (Aligned)', color='orange', linestyle='--', linewidth=2)
        plt.fill_between(range(len(user_warped)), user_warped, pro_warped, color='red', alpha=0.15, label='Error Gap')

        plt.title(f'Swimming Form DTW Alignment (AI Score: {score:.1f}/100)')
        plt.xlabel('Aligned Time Steps')
        plt.ylabel('Elbow Angle (Degrees)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# ================= 交互菜单启动 =================
if __name__ == "__main__":
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

        target_video = input(f"请输入测试视频路径 [默认: {default_video}]: ").strip()
        if not target_video: target_video = default_video

        if not os.path.exists(target_video):
            print(f"错误：找不到文件 '{target_video}'，请检查路径。")
            continue

        use_enhance = input("是否启用 CLAHE 图像增强 (针对水花遮挡/模糊)? (y/n) [默认: n]: ").strip().lower() == 'y'

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