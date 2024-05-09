

"""this fie using libraries"""
from cv2 import VideoCapture
import cv2
import numpy as np
import imutils

PROTOTEXT = "models/SSD_MobileNet_prototxt.txt"
CAFEE = "models/SSD_MobileNet.caffemodel"

TRESH = 0.2

CAFEE_MODEL = cv2.dnn.readNetFromCaffe(PROTOTEXT, CAFEE)

labels = ["background", "aeroplane", "bicycle", "bird",
"boat","bottle", "bus", "car", "cat", "chair", "cow", 
"diningtable","dog", "horse", "motorbike", "person", "pottedplant", 
"sheep","sofa", "train", "tvmonitor"]

colors = np.random.uniform(0, 255, size=(len(labels), 3))

cam = VideoCapture(0)

while True:
    _, image = cam.read()

    image = imutils.resize(image, width=500)

    (h, w) = image.shape[:2]

    BLOB = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    CAFEE_MODEL.setInput(BLOB)
    point = CAFEE_MODEL.forward()

    for i in np.arange(0, point.shape[2]):
        confe = point[0, 0, i, 2]

        if confe > TRESH:

            idx = int(point[0, 0, i, 1])

            box = point[0, 0, i, 3:7] * np.array([w, h, w, h])

            (startx, starty, endx, endy) = box.astype('int')
            label = f"{labels[idx]} | {float(confe * 100)}"
            cv2.rectangle(image, (startx, starty), (endx, endy), colors[idx], 2)

            y = starty - 15 if starty -15 > 15 else starty + 15

            cv2.putText(image, label, (startx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

    cv2.imshow("Frame", image)

    if cv2.waitKey(1) & 0xFF == ord('z'):
        break

cam.release()
cv2.destroyAllWindows()
