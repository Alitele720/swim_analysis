import cv2
import numpy as np
from ultralytics import YOLO


# ==========================================
# 辅助函数：计算关节角度
# ==========================================
def calculate_angle(a, b, c):
    """
    计算三点(a, b, c)形成的角度，b为顶点。
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


# ==========================================
# 主程序：实时姿态分析与视觉反馈
# ==========================================
def process_swimming_analysis(video_path, output_path='output_analysis.mp4'):
    print("正在加载 YOLOv8x-pose 模型 (高精度版)...")
    model = YOLO('yolov8x-pose.pt')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # 获取视频参数
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_count = 0
    print("开始处理视频... (按 Q 键退出实时预览)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # === 关键改进 1: 图像增强 ===
        # 提高对比度，帮助模型在水花中识别肢体
        frame_enhanced = cv2.convertScaleAbs(frame, alpha=1.5, beta=10)

        # === 关键改进 2: 使用增强后的图像进行推理 ===
        results = model(frame_enhanced, verbose=False)

        # 在原图(增强后)上绘制骨架
        annotated_frame = results[0].plot()

        # 解析关键点
        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            keypoints = results[0].keypoints.data[0].cpu().numpy()

            # 假设追踪【右臂】: 6(肩), 8(肘), 10(腕)
            # 如果是左臂请改为: 5, 7, 9
            idx_shoulder, idx_elbow, idx_wrist = 6, 8, 10

            # 获取坐标和置信度
            shoulder = keypoints[idx_shoulder][:2]
            elbow = keypoints[idx_elbow][:2]
            wrist = keypoints[idx_wrist][:2]

            conf_s = keypoints[idx_shoulder][2]
            conf_e = keypoints[idx_elbow][2]
            conf_w = keypoints[idx_wrist][2]

            # 只有当三个点都清晰可见时，才计算角度
            if conf_s > 0.5 and conf_e > 0.5 and conf_w > 0.5:
                # 计算肘部角度
                elbow_angle = calculate_angle(shoulder, elbow, wrist)

                # === 视觉反馈 ===
                # 1. 显示角度数值
                cv2.putText(annotated_frame, f"{int(elbow_angle)} deg",
                            (int(elbow[0]), int(elbow[1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # 2. 姿态判断逻辑 (以自由泳抱水为例)
                # 如果角度过大(接近180度)，说明是"直臂划水"(效率低)
                # 如果角度在 90-150 之间，通常是较好的"高肘抱水"
                if elbow_angle > 160:
                    status = "WARN: Straight Arm!"
                    color = (0, 0, 255)  # Red
                else:
                    status = "Good: High Elbow"
                    color = (0, 255, 0)  # Green

                cv2.putText(annotated_frame, status, (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 显示帧号
        cv2.putText(annotated_frame, f"Frame: {frame_count}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 写入和显示
        out.write(annotated_frame)
        cv2.imshow('Swimming Posture Analysis', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"分析完成，视频已保存至 {output_path}")


if __name__ == "__main__":
    video_file = "test.mp4"  # 你的视频路径
    process_swimming_analysis(video_file)