import cv2 as cv
import numpy as np
import time
import HandTrackingModule as htm
import os

##################
brushThickness = 15

##################


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
    image = rescaleImage(image, 125, 1280)
    overlayList.append(image)
# print(len(overlayList))

header = overlayList[0]

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.HandDetector(detectionCon = 0.85) # Higher confidence to avoid mistakes

drawColor = (255, 0, 255)
xp, yp = 0, 0
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
            
            print("Selection Mode")

            # Checking for the click
            if y1 < 125:
                if (200<x1<400):
                    header = overlayList[0]
                    drawColor = (0, 0, 255)
                elif(400<x1<600):
                    header = overlayList[1]
                    drawColor = (0, 255, 255)
                elif(600<x1<800):
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif(800<x1<1000):
                    header = overlayList[3]
                    drawColor = (255, 255, 255)

            cv.rectangle(img, (x1, y1+15), (x2, y2+15), drawColor, -1)

        # 5. If drawing mode - index finger is up
        if (fingers[1] & fingers[2]== False):
            cv.circle(img, (x1, y1), 15, drawColor, -1)
            print("Drawing Mode")

            if (xp ==0 & yp == 0):
                xp, yp = x1, y1
            
            cv.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1


    # Setting the header image
    img[0:125, 0:1280] = header

    cv.imshow("Video", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()