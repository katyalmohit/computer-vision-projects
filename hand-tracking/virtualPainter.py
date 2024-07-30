import cv2 as cv
import numpy as np
import time
import HandTrackingModule as htm
import os


folderPath = "./resources/header-images"

myList = os.listdir(folderPath)
myList = sorted(myList)
print(myList)

overlayList = []

for imPath in myList:
    image = cv.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
# print(len(overlayList))

header = overlayList[0]

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()
    cv.imshow("Video", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()