import cv2 as cv
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

        # Define the drawing specifications for landmarks and connections
        self.handDrawingSpec = self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        self.handConnectionSpec = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS, self.handDrawingSpec, self.handConnectionSpec)
        return img

    def findPosition(self, img, handNum=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 8, (255, 0, 255), -1)
        return lmList

def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)

    detector = HandDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img, draw=True)
        lmList = detector.findPosition(img, draw=True)

        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv.imshow("Video", img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
