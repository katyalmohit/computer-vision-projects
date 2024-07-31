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

    # 1. Find hand landmarks
    # 2. Get the landmarks of tip of index and middle fingers
    # 3. Check which fingers are up
    # 4. Only index finger: Moving mode
    # 5. Convert Coordinates
    # 6. Smoothen Values
    # 7. Move Mouse
    # 8. Both Index and middle fingers are up: Clicking mode
    # 9. Find distance between fingers
    # 10. Click mouse if distance short

    # 11. Frame Rate
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv.putText(img, str(f'FPS:{int(fps)}'), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)


    # 12. Display
    cv.imshow("Video", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()