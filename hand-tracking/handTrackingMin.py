import cv2 as cv
import mediapipe as mp 
import time 

cap = cv.VideoCapture(0)
mpHands = mp.solutions.hands #Assigns hands module of mediapipe to mpHands
hands = mpHands.Hands() #Create instance of Hands

while True:
    success, img = cap.read()

    #Convert to RGB bcz mediapipe expects the input image to be in RGB format
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    #If Hand occurs in a frame it will return values otherwise none
    print(results.multi_hand_landmarks)
    cv.imshow("Video", img)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
