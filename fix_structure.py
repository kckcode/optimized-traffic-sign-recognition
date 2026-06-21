import os
import cv2

folders = [
    "dataset/images/train",
    "dataset/images/val"
]

for folder in folders:
    for file in os.listdir(folder):
        if file.endswith(".ppm"):
            path = os.path.join(folder, file)

            img = cv2.imread(path)
            if img is None:
                continue

            new_path = path.replace(".ppm", ".jpg")
            cv2.imwrite(new_path, img)

            os.remove(path)

print("All images converted to JPG")