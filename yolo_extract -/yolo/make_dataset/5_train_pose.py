import os
import random
import shutil
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCE_IMAGE_DIR = os.path.join(BASE_DIR, "images")
SOURCE_LABEL_DIR = os.path.join(BASE_DIR, "label_editored")

DATASET_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "pose_dataset"))

TRAIN_RATIO = 0.8


def prepare_dataset():

    train_img = os.path.join(DATASET_ROOT, "images/train")
    val_img = os.path.join(DATASET_ROOT, "images/val")
    train_lbl = os.path.join(DATASET_ROOT, "labels/train")
    val_lbl = os.path.join(DATASET_ROOT, "labels/val")

    for path in [train_img, val_img, train_lbl, val_lbl]:
        os.makedirs(path, exist_ok=True)

    images = []

    for f in os.listdir(SOURCE_IMAGE_DIR):
        if f.endswith(".jpg"):
            label_path = os.path.join(SOURCE_LABEL_DIR, f.replace(".jpg", ".txt"))
            if os.path.exists(label_path):
                images.append(f)

    if len(images) == 0:
        print("没有人工标注数据，无法训练")
        exit()

    random.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)
    train_files = images[:split_index]
    val_files = images[split_index:]

    print("人工训练集数量:", len(train_files))
    print("人工验证集数量:", len(val_files))

    for file in train_files:
        shutil.copy(os.path.join(SOURCE_IMAGE_DIR, file), train_img)
        shutil.copy(os.path.join(SOURCE_LABEL_DIR, file.replace(".jpg", ".txt")), train_lbl)

    for file in val_files:
        shutil.copy(os.path.join(SOURCE_IMAGE_DIR, file), val_img)
        shutil.copy(os.path.join(SOURCE_LABEL_DIR, file.replace(".jpg", ".txt")), val_lbl)

    print("数据集划分完成")


def create_yaml():
    yaml_path = os.path.join(DATASET_ROOT, "dataset.yaml")

    content = f"""
path: {DATASET_ROOT}
train: images/train
val: images/val

names:
  0: person

kpt_shape: [17,3]
"""

    with open(yaml_path, "w") as f:
        f.write(content)

    return yaml_path


def train_model(yaml_path):

    model = YOLO("yolov8n-pose.pt")

    model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        name="swim_pose_train"
    )


if __name__ == "__main__":

    prepare_dataset()
    yaml_path = create_yaml()
    train_model(yaml_path)