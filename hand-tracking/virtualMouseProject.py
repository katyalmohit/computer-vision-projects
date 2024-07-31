import cv2 as cv
import numpy as np
import time
import HandTrackingModule as htm
import pyautogui


######################
wCam, hCam = 640, 480
######################

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


pTime = 0
detector = htm.HandDetector(maxHands=1)

while True:
    success, img = cap.read()

    cv.imshow("Video", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()