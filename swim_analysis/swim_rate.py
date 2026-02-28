import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def process_swimming_rate(video_path, output_path='output_rate.mp4'):
    model = YOLO('yolov8x-pose.pt')
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # 获取视频FPS，这对于计算时间至关重要
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # === 数据容器 ===
    wrist_y_history = []  # 记录手腕Y坐标的序列
    frame_indices = []  # 记录对应的帧号

    frame_count = 0
    print("正在分析动作波形...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # =========== 新增：图像增强 ===========
        # 方法A：简单对比度增强 (最推荐，计算量小)
        # alpha 是对比度 (1.0-3.0), beta 是亮度 (0-100)

        frame_enhanced = cv2.convertScaleAbs(frame, alpha=1.5, beta=10)
        frame_count += 1
        # results = model(frame_enhanced, conf=0.2, verbose=False)
        results = model(frame_enhanced, verbose=False)

        # 绘制骨架 (保留上一阶段的视觉效果)
        annotated_frame = results[0].plot()

        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            keypoints = results[0].keypoints.data[0].cpu().numpy()

            # 假设我们追踪【右腕 (Right Wrist)】 -> 索引 10
            # 如果是左侧视角，请改为索引 9 (Left Wrist)
            # 注意：OpenCV中 Y 坐标向下增大，所以"高处"其实是 Y 值较小
            right_wrist_y = keypoints[10][1]
            conf = keypoints[10][2]

            # 只记录置信度高的数据
            if conf > 0.5:
                wrist_y_history.append(right_wrist_y)
                frame_indices.append(frame_count)

                # 实时在画面上画个小点，显示当前追踪位置
                cv2.circle(annotated_frame, (int(keypoints[10][0]), int(right_wrist_y)),
                           10, (0, 0, 255), -1)

        cv2.putText(annotated_frame, f"Analyzing Frame: {frame_count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(annotated_frame)
        cv2.imshow('Stroke Analysis', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # # === 核心算法：划频分析 ===
    # print("视频处理完毕，开始计算划频参数...")
    #
    # if len(wrist_y_history) == 0:
    #     print("未检测到手腕数据，请检查视频或关键点索引。")
    #     return
    #
    # # 1. 数据平滑 (可选，防止手抖导致误判)
    # # 这里把 Y 坐标取反，因为图像坐标系 Y 是向下的，取反后波峰才代表"高点"
    # y_data = -1 * np.array(wrist_y_history)
    #
    # # 2. 寻找波峰 (Find Peaks)
    # # distance: 两个波峰之间至少间隔多少帧 (防止微小抖动被算作两次划水)
    # # 假设划水最快也就 1秒1次 (60SPM)，如果FPS是30，那distance设为15比较安全
    # min_distance = int(fps * 0.5)
    # peaks, _ = find_peaks(y_data, distance=min_distance)
    #
    # # 3. 计算统计数据
    # num_strokes = len(peaks)
    # duration_sec = frame_count / fps
    # spm = (num_strokes / duration_sec) * 60 if duration_sec > 0 else 0
    #
    # print(f"========================================")
    # print(f"检测到的划水次数: {num_strokes} 次")
    # print(f"视频总时长: {duration_sec:.2f} 秒")
    # print(f"平均划频 (SPM): {spm:.2f} 次/分")
    # print(f"========================================")
    #
    # # === 可视化图表生成 ===
    # plt.figure(figsize=(12, 6))
    # plt.plot(frame_indices, y_data, label='Wrist Vertical Movement')
    # plt.plot(np.array(frame_indices)[peaks], y_data[peaks], "x", color='red', label='Detected Stroke (Peak)')
    # plt.title(f'Swimming Stroke Analysis (SPM: {spm:.1f})')
    # plt.xlabel('Frame Index')
    # plt.ylabel('Wrist Vertical Position (Inverted)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()



    # # === 核心算法：划频分析 (优化版 V2.1) ===
    # print("视频处理完毕，开始计算划频参数...")
    #
    # if len(wrist_y_history) == 0:
    #     print("未检测到手腕数据。")
    #     return
    #
    # # 1. 数据准备
    # y_data = -1 * np.array(wrist_y_history)
    #
    # # 2. 平滑处理 (Smoothing) - 关键步骤！
    # # 使用移动平均窗口来消除 YOLO 的抖动噪音
    # window_size = 5  # 窗口越大曲线越平滑，但太大会导致滞后
    # y_smoothed = np.convolve(y_data, np.ones(window_size) / window_size, mode='valid')
    # # 调整帧索引以匹配平滑后的数据长度
    # frame_indices_smoothed = frame_indices[window_size - 1:]
    #
    # # 3. 智能波峰检测
    # # height: 只有高度超过这个值的峰才算数。
    # # 观察你的图，有效波峰大约在 -200 以上。我们将阈值设为数据的平均值，或者固定值。
    # # 这里我们动态计算：取最大值和最小值的中间偏上一点作为门槛。
    # data_range = np.max(y_smoothed) - np.min(y_smoothed)
    # height_threshold = np.min(y_smoothed) + (data_range * 0.6)  # 只保留最高的 40% 的波峰
    #
    # # distance: 两个动作之间的最小间隔帧数 (防止一次划水算两次)
    # min_distance = int(fps * 0.8)
    #
    # peaks, _ = find_peaks(y_smoothed, height=height_threshold, distance=min_distance)
    #
    # # 4. 计算统计数据
    # num_strokes = len(peaks)
    # duration_sec = frame_count / fps
    # spm = (num_strokes / duration_sec) * 60 if duration_sec > 0 else 0
    #
    # print(f"========================================")
    # print(f"【优化后结果】")
    # print(f"高度阈值设定为: {height_threshold:.1f} (低于此线的抖动被忽略)")
    # print(f"检测到的划水次数: {num_strokes} 次")
    # print(f"修正后的划频 (SPM): {spm:.2f} 次/分")
    # print(f"========================================")
    #
    # # === 可视化图表生成 ===
    # plt.figure(figsize=(12, 6))
    #
    # # 画出原始数据（浅灰色，作为对比）
    # plt.plot(frame_indices, y_data, color='lightgray', label='Raw Data (Noisy)', alpha=0.5)
    #
    # # 画出平滑后的数据（蓝色）
    # plt.plot(frame_indices_smoothed, y_smoothed, label='Smoothed Data')
    #
    # # 画出阈值线（绿色虚线）
    # plt.axhline(y=height_threshold, color='green', linestyle='--', label='Height Threshold')
    #
    # # 画出波峰
    # plt.plot(np.array(frame_indices_smoothed)[peaks], y_smoothed[peaks], "x", color='red', markersize=10,
    #          label='Valid Stroke')
    #
    # plt.title(f'Optimized Stroke Analysis (SPM: {spm:.1f})')
    # plt.xlabel('Frame Index')
    # plt.ylabel('Wrist Vertical Position (Inverted)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # === 核心算法：划频分析 (V3.0 自动补全版) ===
    print("视频处理完毕，开始进行【数据修复】与分析...")

    # 1. 检查数据量
    if len(wrist_y_history) < 10:
        print("错误：采集到的有效数据太少，无法分析。请尝试降低置信度阈值或更换模型。")
        exit()

    # 转换数据格式
    valid_frames = np.array(frame_indices)  # 有数据的帧号 (比如: 1, 2, 5, 6, 9...)
    valid_y = -1 * np.array(wrist_y_history)  # 对应的 Y 值

    # 2. 数据插值 (Interpolation) - 修复断连的核心步骤！
    # 创建一个从第一帧到最后一帧的连续序列
    full_frames = np.arange(valid_frames[0], valid_frames[-1] + 1)

    # 使用线性插值填补那些 YOLO 没检测到的帧
    # np.interp(需要填补的x, 已知的x, 已知的y)
    full_y_interp = np.interp(full_frames, valid_frames, valid_y)

    # 3. 平滑处理 (Smoothing)
    # 对修复后的完整曲线进行平滑
    window_size = 15  # 稍微调大窗口，让曲线更圆润
    # 使用更高级的汉宁窗平滑，比简单的平均更顺滑
    window = np.hanning(window_size)
    window = window / window.sum()
    y_smoothed = np.convolve(full_y_interp, window, mode='same')

    # 4. 智能波峰检测
    # 动态计算阈值：取最大值和最小值的中间位置，再偏下一点
    data_min = np.min(y_smoothed)
    data_max = np.max(y_smoothed)
    height_threshold = data_min + (data_max - data_min) * 0.4  # 阈值设为波幅的 40% 处

    # 限制两个波峰间的最小距离 (假设最快划频 60 SPM -> 1秒1次 -> fps帧)
    min_distance = int(fps * 0.6)

    peaks, _ = find_peaks(y_smoothed, height=height_threshold, distance=min_distance)

    # 5. 计算统计数据
    num_strokes = len(peaks)
    # 使用实际的时间跨度来计算，而不是总帧数
    duration_frames = full_frames[-1] - full_frames[0]
    duration_sec = duration_frames / fps
    spm = (num_strokes / duration_sec) * 60 if duration_sec > 0 else 0

    print(f"========================================")
    print(f"【V3.0 分析结果】")
    print(f"数据修复: 原始 {len(valid_frames)} 帧 -> 补全后 {len(full_frames)} 帧")
    print(f"检测到的划水次数: {num_strokes} 次")
    print(f"计算出的划频 (SPM): {spm:.1f} 次/分")
    print(f"========================================")

    # === 可视化图表生成 ===
    plt.figure(figsize=(12, 6))

    # 画出原始检测点 (灰色散点，代表 YOLO 实际捕捉到的位置)
    plt.scatter(valid_frames, valid_y, color='gray', s=10, label='Raw Detections', alpha=0.4)

    # 画出插值补全路径 (橙色虚线，代表数学修复的路径)
    plt.plot(full_frames, full_y_interp, color='orange', linestyle='--', label='Interpolated Path', alpha=0.6)

    # 画出最终平滑曲线 (蓝色实线，主要参考线)
    plt.plot(full_frames, y_smoothed, label='Smoothed Curve', color='#1f77b4', linewidth=2)

    # 画出阈值判定线 (绿色虚线)
    plt.axhline(y=height_threshold, color='green', linestyle='--', label='Height Threshold')

    # 画出识别到的划水时刻 (红色 X)
    plt.plot(full_frames[peaks], y_smoothed[peaks], "x", color='red', markersize=12, markeredgewidth=3,
             label='Stroke Peak')

    # 设置图表标题和坐标轴 (英文，防止乱码)
    plt.title(f'Swimming Stroke Analysis (SPM: {spm:.1f})', fontsize=14)
    plt.xlabel('Frame Index (Time)', fontsize=12)
    plt.ylabel('Wrist Vertical Position', fontsize=12)

    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    video_file = "test.mp4"  # 记得改名
    process_swimming_rate(video_file)