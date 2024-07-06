import cv2 as cv
import mediapipe as mp 
import time 

cap = cv.VideoCapture(0)
mpHands = mp.solutions.hands #Assigns hands module of mediapipe to mpHands
hands = mpHands.Hands() #Create instance of Hands

while True:
    success, img = cap.read()
    cv.imshow("Video", img)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
