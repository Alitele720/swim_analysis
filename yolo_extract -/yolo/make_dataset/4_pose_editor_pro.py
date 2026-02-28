import cv2
import os
import glob

# ===================== 路径配置 =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCE_DIR = os.path.join(BASE_DIR, "images")
ORIGIN_LABEL_DIR = os.path.join(BASE_DIR, "labels")            # 自动检测标签
SAVE_LABEL_DIR = os.path.join(BASE_DIR, "label_editored")     # 人工标签
SAVE_IMG_DIR = os.path.join(BASE_DIR, "images_editor")

os.makedirs(SAVE_LABEL_DIR, exist_ok=True)
os.makedirs(SAVE_IMG_DIR, exist_ok=True)
# ===================================================

KPT_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_sho", "r_sho", "l_elb", "r_elb",
    "l_wri", "r_wri", "l_hip", "r_hip",
    "l_kne", "r_kne", "l_ank", "r_ank"
]

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# ===================== 读取有标签图片 =====================
image_files = []
for img_path in glob.glob(os.path.join(SOURCE_DIR, "*.jpg")):
    auto_label = os.path.join(
        ORIGIN_LABEL_DIR,
        os.path.basename(img_path).replace(".jpg", ".txt")
    )
    if os.path.exists(auto_label):
        image_files.append(img_path)

image_files = sorted(image_files)

if len(image_files) == 0:
    print("没有可编辑图片")
    exit()

index = 0
mode = "keypoint"  # keypoint / bbox
selected_point = None
selected_corner = None
points = []
bbox = None


# ===================== 加载图像与标签 =====================
def load_image(idx):
    global img, h, w, points, bbox

    image_path = image_files[idx]
    filename = os.path.basename(image_path)

    edited_label = os.path.join(SAVE_LABEL_DIR, filename.replace(".jpg", ".txt"))
    auto_label = os.path.join(ORIGIN_LABEL_DIR, filename.replace(".jpg", ".txt"))

    # 优先读取人工标签
    label_path = edited_label if os.path.exists(edited_label) else auto_label

    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    points.clear()

    with open(label_path, "r") as f:
        data = list(map(float, f.readline().split()))

    bbox = data[1:5]
    kpts = data[5:]

    for i in range(0, len(kpts), 3):
        x = int(kpts[i] * w)
        y = int(kpts[i + 1] * h)
        v = int(kpts[i + 2])
        points.append([x, y, v])


# ===================== 保存标签 + 可视化图 =====================
def save_label_and_image():
    global bbox, points

    image_path = image_files[index]
    filename = os.path.basename(image_path)

    label_path = os.path.join(
        SAVE_LABEL_DIR,
        filename.replace(".jpg", ".txt")
    )

    new_line = [0] + bbox

    for x, y, v in points:
        norm_x = max(0.0, min(1.0, x / w))
        norm_y = max(0.0, min(1.0, y / h))
        new_line.extend([norm_x, norm_y, v])

    with open(label_path, "w") as f:
        f.write(" ".join(map(str, new_line)))

    save_img = img.copy()
    draw_bbox(save_img)
    draw_keypoints(save_img)

    save_path = os.path.join(SAVE_IMG_DIR, filename)
    cv2.imwrite(save_path, save_img)

    print("已保存人工标签:", filename)


# ===================== 绘制函数 =====================
def draw_bbox(temp):
    if bbox is None:
        return
    xc, yc, bw, bh = bbox
    x1 = int((xc - bw / 2) * w)
    y1 = int((yc - bh / 2) * h)
    x2 = int((xc + bw / 2) * w)
    y2 = int((yc + bh / 2) * h)

    cv2.rectangle(temp, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # bbox 四个角点
    for (cx, cy) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        cv2.circle(temp, (cx, cy), 6, (0, 255, 255), -1)


def draw_keypoints(temp):
    # 骨架
    for (i, j) in SKELETON:
        if i < len(points) and j < len(points):
            if points[i][2] > 0 and points[j][2] > 0:
                cv2.line(temp, (points[i][0], points[i][1]),
                         (points[j][0], points[j][1]), (0, 255, 0), 2)

    # 关键点
    for i, (x, y, v) in enumerate(points):
        if v > 0:
            cv2.circle(temp, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(temp, KPT_NAMES[i], (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            cv2.circle(temp, (x, y), 5, (128, 128, 128), 2)


# ===================== 鼠标事件 =====================
def mouse_callback(event, x, y, flags, param):
    global selected_point, selected_corner, bbox

    xc, yc, bw, bh = bbox
    x1 = int((xc - bw / 2) * w)
    y1 = int((yc - bh / 2) * h)
    x2 = int((xc + bw / 2) * w)
    y2 = int((yc + bh / 2) * h)

    if mode == "keypoint":

        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (px, py, pv) in enumerate(points):
                if abs(px - x) < 8 and abs(py - y) < 8:
                    selected_point = i

        elif event == cv2.EVENT_RBUTTONDOWN:
            for i, (px, py, pv) in enumerate(points):
                if abs(px - x) < 8 and abs(py - y) < 8:
                    points[i][2] = 2 if pv == 0 else 0

        elif event == cv2.EVENT_MOUSEMOVE and selected_point is not None:
            points[selected_point][0] = x
            points[selected_point][1] = y

        elif event == cv2.EVENT_LBUTTONUP:
            selected_point = None

    else:  # bbox 模式

        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (cx, cy) in enumerate(corners):
                if abs(cx - x) < 10 and abs(cy - y) < 10:
                    selected_corner = i

        elif event == cv2.EVENT_MOUSEMOVE and selected_corner is not None:

            if selected_corner == 0:
                x1, y1 = x, y
            elif selected_corner == 1:
                x2, y1 = x, y
            elif selected_corner == 2:
                x1, y2 = x, y
            elif selected_corner == 3:
                x2, y2 = x, y

            bbox = [
                (x1 + x2) / (2 * w),
                (y1 + y2) / (2 * h),
                abs(x2 - x1) / w,
                abs(y2 - y1) / h
            ]

        elif event == cv2.EVENT_LBUTTONUP:
            selected_corner = None


# ===================== 主程序 =====================
cv2.namedWindow("Pose Editor Pro")
cv2.setMouseCallback("Pose Editor Pro", mouse_callback)

load_image(index)

while True:
    temp = img.copy()
    draw_bbox(temp)
    draw_keypoints(temp)

    cv2.putText(temp, f"{index + 1}/{len(image_files)}  Mode:{mode}  (Tab: Switch Mode)",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2)

    cv2.imshow("Pose Editor Pro", temp)
    key = cv2.waitKey(1)

    if key == 9:  # Tab 键
        mode = "bbox" if mode == "keypoint" else "keypoint"

    elif key == ord('s'):
        save_label_and_image()

    elif key == ord('d'):
        index = (index + 1) % len(image_files)
        load_image(index)

    elif key == ord('a'):
        index = (index - 1) % len(image_files)
        load_image(index)

    elif key == 27:
        break

cv2.destroyAllWindows()