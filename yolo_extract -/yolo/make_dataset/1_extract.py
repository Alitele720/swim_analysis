import cv2
from tqdm import tqdm
import sys
import os
import random

# ================= 路径处理 =================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(ROOT_DIR)

from preprocess import preprocess_frame

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= 参数 =================
FRAMES_PER_SECOND = 5        # 每秒抽几帧
AUGMENT_PROBABILITY = 0.5    #  50% 概率做预处理（可改）

VIDEO_DIR = os.path.join(BASE_DIR, "videos")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
# ==========================================

os.makedirs(IMAGE_DIR, exist_ok=True)

video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]

for video_name in video_files:
    video_path = os.path.join(VIDEO_DIR, video_name)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ 无法打开视频：{video_name}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    save_interval_sec = 1.0 / FRAMES_PER_SECOND
    next_save_time = 0.0

    video_base = os.path.splitext(video_name)[0]
    image_index = 1

    print(f"\n 处理 {video_name} | fps={fps:.2f}")

    with tqdm(total=total_frames, unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            if current_time >= next_save_time:
                img_name = f"{video_base}_{image_index:04d}.jpg"
                img_path = os.path.join(IMAGE_DIR, img_name)

                # ==========================
                # 随机一半做预处理
                # ==========================
                if random.random() < AUGMENT_PROBABILITY:
                    frame_to_save = preprocess_frame(
                        frame,
                        training=False,     # ⚠ 不再让 preprocess 自己随机
                        noise_robust=False
                    )
                else:
                    frame_to_save = frame  # 原图直接保存

                cv2.imwrite(img_path, frame_to_save)

                image_index += 1
                next_save_time += save_interval_sec

            pbar.update(1)

    cap.release()

print("\n✅ 所有视频抽帧完成")