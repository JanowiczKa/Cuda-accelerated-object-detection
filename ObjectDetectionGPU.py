import cv2
from cv2 import cuda
import numpy as np
import time
import os
import zlib

from sympy import total_degree

#OPTIONS

#True = enable cuda acceleration
gpuAcceleration = True 

#if you'd like to use the webcam replace with 0
capture = cv2.VideoCapture("./images/cars-busy-streets-city-traffic.mp4") #"./images/cars-busy-streets-city-traffic.mp4"

#minimum confience needed for boxes to appear
confidenceMin = 0.5 



imgSize = 416
classes = open('coco.names').read().strip().split('\n')
np.random.seed(42)

colors = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet('yolov3.cfg', 'yolov3.weights')

if (gpuAcceleration):
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

capture.set(3,imgSize)
capture.set(4,imgSize)

totalDelay = 0
totalFrames = 0
while True:
    success, img = capture.read()

    if (not success):
        break

    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (imgSize, imgSize), swapRB=True, crop=False)
    
    net.setInput(blob)
    prevTime = time.time()

    outputs = net.forward(ln)

    currTime = time.time() - prevTime

    delayTime = round(currTime*1000, 1)

    if (totalFrames == 1):
        totalDelay = 0

    totalDelay += delayTime
    totalFrames += 1

    #print(str(delayTime) + "ms since last frame.")#ms delay

    outputs = np.vstack(outputs)

    boxBoundies = []
    confidences = []
    classIDs = []

    for output in outputs:

        data = output[5:]
        classID = np.argmax(data)
        detectedConf = data[classID]

        if detectedConf >= confidenceMin:

            height, width = img.shape[:2]
            boundryBoxX, boundryBoxY, boundryBoxW, boundryBoxH = output[:4] * np.array([width, height, width, height])
            confidences.append(float(detectedConf))
            boxBoundies.append([int(boundryBoxX - boundryBoxW // 2), int(boundryBoxY - boundryBoxH // 2), int(boundryBoxW), int(boundryBoxH)])
            classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(boxBoundies, confidences, confidenceMin, confidenceMin - 0.1)
    for i in indices:
        i = i[0]
        box = boxBoundies[i]
                    
        color = [int(c) for c in colors[classIDs[i]]]
        text = "{}: {:.5f}".format(classes[classIDs[i]], confidences[i])

        cv2.rectangle(img, (box[0], box[1]), ((box[0]+box[2]), (box[1]+box[3])), color, 2)
        cv2.putText(img, text, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imshow("Yolo3 Object detection", img)

    if cv2.waitKey(1) & 0xFF == ord("q"): #this is needed to slow down the loop
        break

print("Average delay for this program was " + str(round(totalDelay/totalFrames, 1)) + "ms.")
