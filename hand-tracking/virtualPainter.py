import cv2 as cv
import numpy as np
import time
import HandTrackingModule as htm
import os

##################
brushThickness = 15
eraserThickness = 100   
##################


def rescaleImage(image, height, width):
    width = int(width)
    height = int(height)
    
    dimensions = (width, height) #tuple containing rescaled width and height of image
    
    return cv.resize(image, dimensions, interpolation = cv.INTER_AREA)


folderPath = "./resources/header-images"

myList = os.listdir(folderPath)
myList = sorted(myList)
# print(myList)

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

imgCanvas = np.zeros((720, 1280, 3), np.uint8)

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
        # print(fingers)

        # 4. If selection mode - two fingers are up
        if (fingers[1] & fingers[2]):
            xp, yp = 0, 0
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
                    drawColor = (0, 0, 0)

            cv.rectangle(img, (x1, y1+15), (x2, y2+15), drawColor, -1)

        # 5. If drawing mode - index finger is up
        if (fingers[1] & fingers[2]== False):
            cv.circle(img, (x1, y1), 15, drawColor, -1)
            print("Drawing Mode")

            if (xp ==0 & yp == 0):
                xp, yp = x1, y1
            
            if (drawColor == (255, 255, 255) or drawColor == (0, 0, 0)):
                cv.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)

            else:
                cv.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img, imgInv)
    img = cv.bitwise_or(img, imgCanvas)

    
    # Setting the header image
    img[0:125, 0:1280] = header

    # img = cv.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    cv.imshow("Video", img)
    # cv.imshow("imgInv", imgInv) 
    # cv.imshow("Drawing", imgCanvas)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()