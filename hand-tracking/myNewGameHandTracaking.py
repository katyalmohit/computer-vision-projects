import cv2 as cv
import mediapipe as mpDraw
import time
import HandTrackingModule as htm

pTime = 0 #previous time
cTime = 0 #current time
cap = cv.VideoCapture(0)

detector = handDetector()

while True:
    success, img = cap.read()
    
    img = detector.findHands(img)

    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        print(lmList[4]) #Print the landmarks of a specific handPoint

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    cv.imshow("Video", img)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
