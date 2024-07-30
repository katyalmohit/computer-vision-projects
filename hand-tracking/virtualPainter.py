import cv2 as cv
import numpy as np
import time
import HandTrackingModule as htm
import os

def rescaleImage(image, height, width):
    width = int(width)
    height = int(height)
    
    dimensions = (width, height) #tuple containing rescaled width and height of image
    
    return cv.resize(image, dimensions, interpolation = cv.INTER_AREA)


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
header = rescaleImage(header, 125, 1280)

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()

    # Setting the header image
    img[0:125, 0:1280] = header

    cv.imshow("Video", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()