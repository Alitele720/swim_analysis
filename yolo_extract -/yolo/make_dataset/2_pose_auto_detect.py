import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "images")
SAVE_LABEL_DIR = os.path.join(BASE_DIR, "labels")

os.makedirs(SAVE_LABEL_DIR, exist_ok=True)

model = YOLO("yolov8n-pose.pt")

for file in os.listdir(SOURCE_DIR):
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    image_path = os.path.join(SOURCE_DIR, file)
    results = model(image_path, conf=0.15)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
        continue

    boxes = result.boxes.xywhn.cpu().numpy()

    # 获取归一化坐标和置信度
    kpts_xyn = result.keypoints.xyn.cpu().numpy()[0]
    # 如果模型输出了置信度则获取，否则默认给1.0
    kpts_conf = result.keypoints.conf.cpu().numpy()[0] if result.keypoints.conf is not None else [1.0] * 17

    box = boxes[0]
    txt_line = [0] + box.tolist()

    # 动态写入可见性 v
    for i in range(len(kpts_xyn)):
        x, y = kpts_xyn[i]
        conf = kpts_conf[i]

        # 如果坐标大于0且置信度大于0.5，判定为可见(2)，否则不可见(0)
        v = 2 if (x > 0 or y > 0) and conf > 0.5 else 0
        txt_line.extend([x, y, v])

    label_path = os.path.join(
        SAVE_LABEL_DIR,
        file.replace(".jpg", ".txt").replace(".png", ".txt")
    )

    with open(label_path, "w") as f:
        f.write(" ".join(map(str, txt_line)))

print("检测完成，已处理可见性参数")