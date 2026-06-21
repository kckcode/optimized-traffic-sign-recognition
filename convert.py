import os
import cv2

base_path = "dataset"
gt_file = os.path.join(base_path, "gt.txt")

train_path = os.path.join(base_path, "images/train")
val_path = os.path.join(base_path, "images/val")

label_train = os.path.join(base_path, "labels/train")
label_val = os.path.join(base_path, "labels/val")

os.makedirs(label_train, exist_ok=True)
os.makedirs(label_val, exist_ok=True)

with open(gt_file) as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split(";")
    img_name = parts[0]
    x1, y1, x2, y2 = map(int, parts[1:5])

    if os.path.exists(os.path.join(train_path, img_name)):
        img_path = os.path.join(train_path, img_name)
        label_path = os.path.join(label_train, img_name.replace(".ppm", ".txt"))
    else:
        img_path = os.path.join(val_path, img_name)
        label_path = os.path.join(label_val, img_name.replace(".ppm", ".txt"))

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    width = (x2 - x1) / w
    height = (y2 - y1) / h

    with open(label_path, "w") as f:
        f.write(f"0 {x_center} {y_center} {width} {height}")