import cv2 as cv
import time
import numpy as np
import HandTrackingModule as htm
import math
import alsaaudio
from subprocess import call


minBright, maxBright = 0.1, 1.0

def setBrightness(newBrightness):
    valid = False

    while not valid:
        try:
            brightness = float(newBrightness)
            if minBright <= brightness <= maxBright:
                # Replace 'eDP-1' with your display name, which you can find using 'xrandr --listmonitors'
                call(["xrandr", "--output", "eDP-1", "--brightness", str(brightness)])
                valid = True
            else:
                print(f"Please enter a brightness value between {minBright} and {maxBright}.")
        except ValueError:
            print("Invalid input. Please enter a numeric value for brightness.")

#######################
# Width and height of the feed for webcam
wCam, hCam = 640, 480
#######################

cap = cv.VideoCapture(0)
cap.set(3, wCam) #prop ID at No. 3 is width
cap.set(4, hCam) #prop ID at No. 4 is height

detector = htm.HandDetector() # We can also change the parameters from here

pTime = 0
vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()

    detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:

        # print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2)//2, (y1 + y2)//2

        cv.circle(img, (x1, y1), 8, (255, 0, 255), -1)
        cv.circle(img, (x2, y2), 8, (255, 0, 255), -1)

        cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        cv.circle(img, (cx, cy), 8, (255, 0, 255), -1)

        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)

        # Hand range 10 - 250
        # Volume range 0 - 100

        vol = np.interp(length, [10, 250], [minBright, maxBright])
        volBar = np.interp(length, [10, 250], [400, 150])
        volPer = np.interp(length, [10, 250], [minBright, maxBright])

        # print(length, vol)
        setBrightness(vol)


        if length<50:
            cv.circle(img, (cx, cy), 8, (0, 255, 0), -1)

    cv.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), -1)
    cv.putText(img, f"{int(volPer)}%", (30, 450), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img, f"FPS: {int(fps)}", (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
    cv.imshow("Video", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()