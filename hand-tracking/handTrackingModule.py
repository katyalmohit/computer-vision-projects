import cv2 as cv
import mediapipe as mp 
import time 

class handDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands #Assigns hands module of mediapipe to mpHands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon) #Create instance of Hands
        self.mpDraw = mp.solutions.drawing_utils #Assigns the drawing_utils module from mediapipe

    def findHands(self, img, draw = True):
        #Convert to RGB bcz mediapipe expects the input image to be in RGB format
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        # #If Hand occurs in a frame it will return values otherwise none
        # print(results.multi_hand_landmarks)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if draw:
                    # mpDraw.draw_landmarks(img, handLms) #Draw points on hands
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) #Draw points along with connecting lines
        return img

    
            # for id, lm in enumerate(handLms.landmark):
            #     # print(id, lm) #Print ids and landmarks of hands
            #     h, w, c = img.shape
            #     cx, cy = int(lm.x*w), int(lm.y*h)
            #     print(id, cx, cy)
            #     if (id ==0):
            #         cv.circle(img, (cx, cy), 25, (255, 0, 255), -1)


def main():
    pTime = 0 #previous time
    cTime = 0 #current time
    cap = cv.VideoCapture(0)

    detector = handDetector()

    while True:
        success, img = cap.read()
        
        img = detector.findHands(img)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv.imshow("Video", img)
        if cv.waitKey(20) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()