import cv2 as cv
import mediapipe as mp 
import time 

cap = cv.VideoCapture(0)
mpHands = mp.solutions.hands #Assigns hands module of mediapipe to mpHands
hands = mpHands.Hands() #Create instance of Hands
mpDraw = mp.solutions.drawing_utils #Assigns the drawing_utils module from mediapipe

pTime = 0 #previous time
cTime = 0 #current time

while True:
    success, img = cap.read()

    #Convert to RGB bcz mediapipe expects the input image to be in RGB format
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # #If Hand occurs in a frame it will return values otherwise none
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # mpDraw.draw_landmarks(img, handLms) #Draw points on hands
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) #Draw points along with connecting lines
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    cv.imshow("Video", img)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
