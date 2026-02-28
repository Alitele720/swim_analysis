import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


# ==========================================
# 1. 基础计算函数
# ==========================================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle


# ==========================================
# 2. 特征提取器 (含图像增强 + 数据自动修复)
# ==========================================
def extract_pose_sequence(video_path, description="Video"):
    print(f"Extracting features from: {description} ...")

    # 使用高精度模型
    model = YOLO('yolov8x-pose.pt')
    cap = cv2.VideoCapture(video_path)

    # 原始数据容器
    raw_angles = []
    frame_indices = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        # 为了效率，每2帧取一次样 (可调)
        if frame_count % 2 != 0: continue

        # === 图像增强 ===
        frame_enhanced = cv2.convertScaleAbs(frame, alpha=1.5, beta=10)

        # 推理
        results = model(frame_enhanced, verbose=False)

        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            kp = results[0].keypoints.data[0].cpu().numpy()

            # 假设右侧视角: 6(肩), 8(肘), 10(腕)
            # 如果视频是左侧，需要在这里改成 5, 7, 9
            shoulder = kp[6][:2]
            elbow = kp[8][:2]
            wrist = kp[10][:2]
            conf_check = kp[6][2] * kp[8][2] * kp[10][2]

            if conf_check > 0.5:
                angle = calculate_angle(shoulder, elbow, wrist)
                raw_angles.append(angle)
                frame_indices.append(frame_count)

    cap.release()

    # === 数据修复 (移植自 swim_rate.py) ===
    # 如果数据太少，无法分析
    if len(raw_angles) < 10:
        print(f"Warning: Not enough data detected in {description}.")
        return np.array([])

    # 转换为 numpy 数组
    valid_frames = np.array(frame_indices)
    valid_angles = np.array(raw_angles)

    # 创建连续帧序列 (填补空缺)
    full_frames = np.arange(valid_frames[0], valid_frames[-1] + 1)

    # 线性插值补全断点
    full_angles_interp = np.interp(full_frames, valid_frames, valid_angles)

    # 平滑处理 (消除抖动)
    window_size = 5
    if len(full_angles_interp) > window_size:
        window = np.ones(window_size) / window_size
        full_angles_smooth = np.convolve(full_angles_interp, window, mode='same')
    else:
        full_angles_smooth = full_angles_interp

    # 为了 DTW 格式要求，reshape 成 (N, 1)
    return full_angles_smooth.reshape(-1, 1)


# ==========================================
# 3. DTW 对比主程序
# ==========================================
def compare_swimming_form(user_video, pro_video):
    # 提取特征序列 (此时已经是增强+修复过的数据了)
    seq_user = extract_pose_sequence(user_video, "User Video")
    seq_pro = extract_pose_sequence(pro_video, "Pro Video")

    if len(seq_user) == 0 or len(seq_pro) == 0:
        print("Error: Failed to extract skeleton data.")
        return

    print(f"Data ready. User frames: {len(seq_user)}, Pro frames: {len(seq_pro)}")
    print("Running DTW alignment...")

    # 计算 DTW
    distance, path = fastdtw(seq_user, seq_pro, dist=euclidean)

    # 归一化评分逻辑
    avg_error = distance / len(path)
    # 简单评分公式: 假设平均误差0度是100分，每增加1度扣一点
    score = max(0, 100 - avg_error * 1.5)

    print(f"----------------------------------------")
    print(f"Analysis Result:")
    print(f" - Total Distance: {distance:.2f}")
    print(f" - Avg Error per Frame: {avg_error:.2f} degrees")
    print(f" - AI Form Score: {score:.1f} / 100")
    print(f"----------------------------------------")

    # === 可视化 (英文标签) ===
    plt.figure(figsize=(14, 8))

    # 子图 1: 原始提取的波形 (未对齐)
    plt.subplot(2, 1, 1)
    plt.plot(seq_user, label='User Elbow Angle', color='#1f77b4')
    plt.plot(seq_pro, label='Pro Elbow Angle', color='orange', linestyle='--')
    plt.title(f'Step 1: Extracted Sequences (Before Alignment)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图 2: DTW 对齐后的对比
    # 通过 path 索引将两个序列强行拉齐到同一时间轴
    user_warped = [seq_user[idx_u][0] for idx_u, idx_p in path]
    pro_warped = [seq_pro[idx_p][0] for idx_u, idx_p in path]

    plt.subplot(2, 1, 2)
    plt.plot(user_warped, label='User (Aligned)', color='#1f77b4')
    plt.plot(pro_warped, label='Pro (Aligned)', color='orange', linestyle='--')

    # 填充差异区域 (红色)
    plt.fill_between(range(len(user_warped)), user_warped, pro_warped, color='red', alpha=0.2, label='Form Error')

    plt.title(f'Step 2: DTW Aligned Comparison (Score: {score:.1f})', fontsize=12)
    plt.xlabel('Aligned Time Steps', fontsize=10)
    plt.ylabel('Elbow Angle (Degrees)', fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 在这里修改视频路径
    # 如果没有专业视频，可以先把 test.mp4 复制一份改名为 pro.mp4 来测试代码是否跑通
    user_video_file = "test.mp4"
    pro_video_file = "pro.mp4"

    compare_swimming_form(user_video_file, pro_video_file)