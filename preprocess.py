import cv2
import random

def preprocess_frame(frame,
                     training=False,
                     noise_robust=True):
    """
    统一预处理入口

    training=True  → 训练阶段
    noise_robust=True → 随机增强（50%）
    """

    processed = frame.copy()

    # ==========================
    # 1️⃣ 噪声鲁棒训练（仅训练阶段）
    # ==========================
    if training and noise_robust:
        if random.random() < 0.5:
            # 50% 图像做预处理
            processed = apply_filter(processed)
    else:
        # 推理阶段全部处理
        processed = apply_filter(processed)

    return processed


def apply_filter(frame):
    # 1️⃣ 双边滤波
    frame = cv2.bilateralFilter(frame, 7, 60, 60)

    # 2️⃣ 轻度 CLAHE
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return frame