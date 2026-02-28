import os
import shutil
from ultralytics import YOLO


# ===============================
# 1️⃣ 基础路径
# ===============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGES_EDITOR_DIR = os.path.join(BASE_DIR, "images_editor")
LABELS_EDITOR_DIR = os.path.join(BASE_DIR, "label_editored")

YAML_PATH = os.path.join(BASE_DIR, "pose_dataset.yaml")

RUNS_DIR = os.path.join(BASE_DIR, "runs")
PROJECT_NAME = "pose_train"

SWIM_ANALYSIS_MODEL_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "..", "swim_analysis", "models")
)

os.makedirs(SWIM_ANALYSIS_MODEL_DIR, exist_ok=True)


# ===============================
# 2️⃣ 创建标准 Pose YAML
# ===============================

def create_yaml():
    yaml_content = f"""
path: {BASE_DIR}
train: images_editor
val: images_editor

kpt_shape: [17, 3]

names:
  0: swimmer
"""

    with open(YAML_PATH, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print("✅ pose_dataset.yaml 已更新（仅使用 editor 数据）")


# ===============================
# 3️⃣ 训练
# ===============================

def train_model():

    create_yaml()

    model = YOLO("yolov8n-pose.pt")

    model.train(
        data=YAML_PATH,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        project=RUNS_DIR,
        name=PROJECT_NAME,
        exist_ok=True,
        pretrained=True,
        val=True
    )

    print("✅ 训练完成")

    # ===============================
    # 4️⃣ 复制 best.pt
    # ===============================

    best_model_path = os.path.join(
        RUNS_DIR,
        PROJECT_NAME,
        "weights",
        "best.pt"
    )

    if os.path.exists(best_model_path):
        target_path = os.path.join(SWIM_ANALYSIS_MODEL_DIR, "swimmer_pose_best.pt")
        shutil.copy(best_model_path, target_path)
        print(f"✅ best.pt 已复制到 {target_path}")
    else:
        print("❌ 未找到 best.pt")


# ===============================
# 4️⃣ 主程序
# ===============================

if __name__ == "__main__":
    train_model()