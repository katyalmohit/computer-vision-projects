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

detector = htm.HandDetector(detectionCon = 0.85) # Higher confidence to avoid mistakes

while True:
    # 1. Import image
    success, img = cap.read()
    img = cv.flip(img, 1)

    # 2. Find Hand landmarks
    img = detector.findHands(img, draw = True)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # print(lmList)

        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        print(fingers)

        # 4. If selection mode - two fingers are up
        if (fingers[1] & fingers[2]):
            cv.rectangle(img, (x1, y1+15), (x2, y2+15), (255, 0, 255), -1)
            print("Selection Mode")

        # 5. If drawing mode - index finger is up
        if (fingers[1] & fingers[2]== False):
            cv.circle(img, (x1, y1), 15, (255, 0, 255), -1)
            print("Drawing Mode")


    # Setting the header image
    img[0:125, 0:1280] = header

    cv.imshow("Video", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()