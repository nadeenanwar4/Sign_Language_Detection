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

#Displaying text
imgWhiteText = np.ones((400, 600, 3), np.uint8)*255
font = cv2.FONT_HERSHEY_SIMPLEX
#org = (00, 25)
fontScale = 1
color = (255, 0, 0)
thickness = 2


# Create variables to hold the word and its length
word = ""
word_length = 0
index = 0


labels = ["A", "B", "C", "D", "E", "F", " G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


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

        cv2.putText(imgWhiteText, "Welcome To Sign Language Ditiction...!", (20, 25), font, 0.9, (159, 28, 189), thickness)
        cv2.putText(imgWhiteText, "Instructions:-", (00, 65), font, 0.8, (255,255,0),thickness)
        cv2.putText(imgWhiteText, "Press 'A' to add a letter", (00,105), font, 0.7, (0, 0, 0), thickness)
        cv2.putText(imgWhiteText, ",", (00, 115), font, 0.7, (0, 0, 0), thickness)
        cv2.putText(imgWhiteText, "Press 'S' to enter a space ", (00, 150), font, 0.7, (0, 0, 0), thickness)
        cv2.putText(imgWhiteText, "OR", (00,180), font, 0.7, (0, 0, 0),thickness)
        cv2.putText(imgWhiteText, "Press 'D' to remove a letter ", (00,215), font, 0.7, (0, 0, 0),thickness)
        cv2.putText(imgWhiteText, "STATEMENT: ", (00, 300), font, fontScale, (33, 191, 54), thickness)
        cv2.imshow("Text", imgWhiteText)

    cv2.imshow("image", imgOutput)
    #cv2.waitKey(1)
    key = cv2.waitKey(1)

    if key == ord("A"):
        word += labels[index]
        word_length += 1
        # Display the word
        cv2.putText(imgWhiteText, word, (180, 300), font, fontScale, (159, 28, 189), 1)

    elif key == ord("S"):
         word += " "
         word_length += 1
         # Display the word
         cv2.putText(imgWhiteText, word, (180, 300), font, fontScale, color, 1)


    elif key == ord("D"):
         #Delete the last letter from the word
         if word_length > 0:
            word = word[:-1]
            word_length -= 1
            # Redraw the entire word
            cv2.rectangle(imgWhiteText, (100, 100), (600, 400), (255, 255, 255), -1)
            cv2.putText(imgWhiteText, word, (180, 300), font, fontScale, (214, 21, 34), 1)


