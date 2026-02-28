from ultralytics import YOLO

if __name__ == "__main__":
    # 1. 修改模型：使用检测模型 (yolov8n.pt) 而不是姿态模型 (yolov8n-pose.pt)
    # 除非你的数据集确实包含了骨骼关键点标注，否则请使用 yolov8n.pt
    model = YOLO('yolov8n-pose.pt')

    # 2. 修改路径：指向正确的 yaml 文件位置
    # 假设 train.py 在 swim_analysis 文件夹下，且 yaml 在 dataset/swimmer1/s1.yaml
    yaml_path = r"dataset/swimmer1/s1.yaml"

    # 如果上面路径报错，可以使用绝对路径，例如：
    # yaml_path = r"C:\Users\89111\Desktop\analysis\swim_analysis\dataset\swimmer1\s1.yaml"

    model.train(
        data=yaml_path,
        epochs=30,
        imgsz=640,
        batch=16,  # 建议显式指定 batch size，-1 有时在 Windows 下不稳定
        cache=False,  # 如果内存不够大，建议关闭 cache
        workers=2,  # Windows 下 workers 设置过大容易报错，建议 0-4 之间
        device=0  # 如果有显卡，确保指定 device=0
    )