import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time


cap_object = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
folder = "Data/C"
counter = 0
labels = ["A", "B", "C"]

while True:
    success, img = cap_object.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h/w
        if aspectRatio > 1:
            k = imgSize/h
            calculated_width = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (calculated_width, imgSize))
            imgResizeShape = imgResize.shape
            widthGap = math.ceil((imgSize - calculated_width) / 2)
            imgWhite[:, widthGap:calculated_width+widthGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction, index)

        else:
            k = imgSize/w
            calculated_height = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, calculated_height))
            imgResizeShape = imgResize.shape
            heightGap = math.ceil((imgSize - calculated_height) / 2)
            imgWhite[heightGap:calculated_height+heightGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)

        cv2.rectangle(imgOutput, (x - offset, y - offset-50), (x - offset + 90, y - offset), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y-27), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 4)
        cv2.imshow("imageCrop", imgCrop)
        cv2.imshow("imageWhite", imgWhite)

    cv2.imshow("image", imgOutput)
    cv2.waitKey(1)
