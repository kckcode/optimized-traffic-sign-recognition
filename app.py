import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO

model_yolo = YOLO("runs/detect/train5/weights/best.pt")

results = model_yolo("dataset/images/train/00002.jpg")
img = cv2.imread("dataset/images/train/00002.jpg")

found = False

for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()
    for box in boxes:
        found = True
        x1, y1, x2, y2 = map(int, box)
        cropped = img[y1:y2, x1:x2]
        cv2.imwrite("cropped.jpg", cropped)

if not found:
    print("No traffic sign detected")
    exit()

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img = Image.open("cropped.jpg").convert('RGB')
img = img.resize((32,32))
img = np.array(img)/255.0
img = img.reshape(1,32,32,3).astype('float32')

interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])
pred = np.argmax(output)

class_names = {
    0:"Speed limit 20 km/h",1:"Speed limit 30 km/h",2:"Speed limit 50 km/h",
    3:"Speed limit 60 km/h",4:"Speed limit 70 km/h",5:"Speed limit 80 km/h",
    6:"End of speed limit 80",7:"Speed limit 100 km/h",8:"Speed limit 120 km/h",
    9:"No passing",10:"No passing for vehicles over 3.5 tons",
    11:"Right-of-way at intersection",12:"Priority road",13:"Yield",
    14:"Stop",15:"No vehicles",16:"Vehicles over 3.5 tons prohibited",
    17:"No entry",18:"General caution",19:"Dangerous curve left",
    20:"Dangerous curve right",21:"Double curve",22:"Bumpy road",
    23:"Slippery road",24:"Road narrows on the right",25:"Road work",
    26:"Traffic signals",27:"Pedestrians",28:"Children crossing",
    29:"Bicycles crossing",30:"Beware of ice/snow",31:"Wild animals crossing",
    32:"End of all speed limits",33:"Turn right ahead",34:"Turn left ahead",
    35:"Ahead only",36:"Go straight or right",37:"Go straight or left",
    38:"Keep right",39:"Keep left",40:"Roundabout mandatory",
    41:"End of no passing",42:"End of no passing for vehicles over 3.5 tons"
}

print("Predicted Class:", pred)
print("Prediction:", class_names[pred])
