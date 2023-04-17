import cv2      #openCV library
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

#open the camera
cap_object = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1) #detects only one hand
offset = 20
imgSize = 300
folder = "Data/C"
counter = 0

while True:
    # a method to read the image captured by the camera and stores it in img variable
    # success is a boolean variable that stores true if the img is captured and stored successfully and stores false otherwise
    success, img = cap_object.read()
    #passes the image captured by the camera to capture the hand object from it
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]  #???????????????
        x, y, w, h = hand['bbox']  #boundingBOX of the hand img
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        #img shape is an property which is baiscially a list that contains the dimensions of that img
        imgCropShape = imgCrop.shape

        aspectRatio = h/w
        if aspectRatio > 1:
            k = imgSize/h
            calculated_width = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (calculated_width, imgSize))
            imgResizeShape = imgResize.shape
            widthGap = math.ceil((imgSize - calculated_width) / 2)         #??????
            imgWhite[:, widthGap:calculated_width+widthGap] = imgResize

        else:
            k = imgSize/w
            calculated_height = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, calculated_height))
            imgResizeShape = imgResize.shape
            heightGap = math.ceil((imgSize - calculated_height) / 2)
            imgWhite[heightGap:calculated_height+heightGap, :] = imgResize

        cv2.imshow("imageCrop", imgCrop)
        cv2.imshow("imageWhite", imgWhite)

    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/image_{time.time()}.jpg', imgWhite)
        print(counter)
