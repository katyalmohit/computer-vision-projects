import cv2 as cv
import time
import os
import HandTrackingModule as htm

#######################
# Width and height of the feed for webcam
wCam, hCam = 640, 480
#######################

cap = cv.VideoCapture(0)
cap.set(3, wCam) #prop ID at No. 3 is width
cap.set(4, hCam) #prop ID at No. 4 is height

folderPath = "./images"
myList = os.listdir(folderPath)
myList = sorted(myList)
print(myList)
overlayList = []

for imPath in myList:
    image = cv.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

pTime = 0
detector = htm.HandDetector()
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)
    
    if len(lmList) != 0:
        fingers = []

        # Thumb
        if (lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]):
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if (lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]):
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)

        ## Finger count
        totalFingers = fingers.count(1)
        print(totalFingers)

        h, w, c = overlayList[totalFingers].shape
        img[0:h, 0:w] = overlayList[totalFingers]

        cv.rectangle(img, (20, 225), (170, 425), (0, 255, 0), -1)
        cv.putText(img, str(totalFingers), (45, 375), cv.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    cTime = time.time()
    fps =  1/(cTime - pTime)
    pTime = cTime

    cv.putText(img, f"FPS: {int(fps)}", (400, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

    cv.imshow("Video", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()