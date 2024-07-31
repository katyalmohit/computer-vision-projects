import cv2 as cv
import numpy as np
import time
import HandTrackingModule as htm
import pyautogui


######################
wCam, hCam = 640, 480
frameR = 100 # Frame Reduction
smoothening = 5
######################

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


pTime = 0
pLocX, pLocY = 0, 0 #Previous location
cLocX, cLocY = 0, 0 #Current location

detector = htm.HandDetector(maxHands=1)

wScr, hScr = pyautogui.size()
# wScr, hScr = 1920, 1080
# print(wScr, hScr)
while True:
    success, img = cap.read()
    img = cv.flip(img, 1)
    # 1. Find hand landmarks
    img = detector.findHands(img)
    lmList= detector.findPosition(img, draw = False)

    # 2. Get the landmarks of tip of index and middle fingers
    cv.rectangle(img, (frameR, frameR), (wCam-frameR, hCam-frameR), (255, 0, 255), 2)
    if len(lmList) !=0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)


        # 4. Only index finger: Moving mode
        if fingers[1]==1 and fingers[2]==0:

            # 5. Convert Coordinates
            
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

            # 6. Smoothen Values
            cLocX = pLocX + (x3 - pLocX) /smoothening
            cLocY = pLocY + (y3 - pLocY) /smoothening

            # 7. Move Mouse
            pyautogui.moveTo(cLocX, cLocY)
            cv.circle(img, (x1, y1), 15, (255, 0, 255), -1)
            pLocX, pLocY = cLocX, cLocY


        # 8. Both Index and middle fingers are up: Clicking mode
        if fingers[1]==1 and fingers[2]==1:
            # 9. Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)

            # 10. Click mouse if distance short
            if length<40:
                cv.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), -1)
                pyautogui.click()


    
    

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