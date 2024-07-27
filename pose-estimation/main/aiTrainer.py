import cv2 as cv
import time
import numpy as np
import PoseModule as pm


cap = cv.VideoCapture('../videos/dumbbells.mp4')
# cap = cv.VideoCapture(0)

count = 0
dir = 0         # 0 denotes going up, 1 denotes going down
pTime = 0

detector = pm.PoseDetector()

while True:
    success, img = cap.read()
    img = cv.resize(img, (900, 500))

    img = detector.findPose(img, draw = False)
    
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    if len(lmList) != 0:
        # Left Arm
        angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (30, 130), (0, 100))
        bar = np.interp(angle, (30, 130), (100, 400)) # Mapping the values of angle to bar values

        # print(angle, per)

        # Check for the dumbbell curls
        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if dir==1:
                count+=0.5
                dir = 0
                
        if per == 0:
            color = (0, 255, 0)
            if dir==0:
                count+=0.5
                dir = 1
        # print(count)

        # Draw bar
        cv.rectangle(img, (800, 100), (850, 400), color, 3)
        cv.rectangle(img, (800, int(bar)), (850, 400), color, -1)
        cv.putText(img, f'{str(100 - int(per))}%', (760, 70), cv.FONT_HERSHEY_PLAIN, 3, color, 3)

        cv.rectangle(img, (0, 450), (250, 500), (0, 255, 0), -1)
        cv.putText(img, f'Count: {str(int(count))}', (10, 490), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 5)

        ## If we want to display the count in decimals
        # cv.putText(img, str(count), (50, 100), cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)


        # # Right Arm
        # detector.findAngle(img, 12, 14, 16)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img, str(f'FPS: {int(fps)}'), (10, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 5)
    cv.imshow("Video", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# cap.release()
cv.destroyAllWindows()