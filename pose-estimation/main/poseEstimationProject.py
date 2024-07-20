import cv2 as cv
import time
import poseModule as pm

# capture = cv.VideoCapture('../videos/walking.mp4')
capture = cv.VideoCapture('../videos/dance2.mp4')
# capture = cv.VideoCapture('../videos/dance3.mp4')
# capture = cv.VideoCapture('../videos/jumping.mp4')
# capture = cv.VideoCapture('../videos/skipping.mp4')

# capture = cv.VideoCapture(0)
detector = pm.poseDetector()

pTime = 0
while True:
    isTrue, frame = capture.read()
    img = detector.findPose(frame, draw=True)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) !=0  :
        print(lmList[14])

        # Track a specific joint/point
        cv.circle(img, (lmList[14][1], lmList[14][2]), 10, (255, 0, 0), -1)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # cv.imshow('Original video',frame)
    cv.imshow('Resized Video',img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv.destroyAllWindows()