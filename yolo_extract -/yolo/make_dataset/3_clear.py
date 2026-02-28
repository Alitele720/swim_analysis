import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_DIR = os.path.join(BASE_DIR, "labels")

for file in os.listdir(LABEL_DIR):
    path = os.path.join(LABEL_DIR, file)
    if os.path.getsize(path) == 0:
        os.remove(path)
        print("删除空文件:", file)
