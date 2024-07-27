import cv2 as cv
import time
import numpy as np
import PoseModule as pm

#defining a function which takes first parameter as the original frame 
#and second parameter a scale to rescale the video
#here, scale is 0.75 set by default which can be manipulated
def rescaleFrame(frame, scale =0.25):
    #this method works on images, videos and live
    width = int(frame.shape[1]*scale) # Here, 1 is used for original width of the video
    height = int(frame.shape[0]*scale) # Here, '0' is used for original height of video
    
    dimensions = (width, height) # defining the dimensions using new width and height using a TUPLE
    
    return cv.resize(frame, dimensions, interpolation= cv.INTER_AREA) #returning the rescaled values



cap = cv.VideoCapture('../videos/dumbbells.mp4')

count = 0
dir = 0         # 0 denotes going up, 1 denotes going down

detector = pm.PoseDetector()

while True:
    success, img = cap.read()
    img = rescaleFrame(img, scale = 0.9)

    # img = cv.imread('../videos/dips2.jpg')
    # img = rescaleFrame(img, scale = 0.5)

    img = detector.findPose(img, draw = False)
    
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    if len(lmList) != 0:
        # Left Arm
        angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (30, 130), (0, 100))
        # print(angle, per)

        # Check for the dumbbell curls
        if per == 100:
            if dir==1:
                count+=0.5
                dir = 0
        if per == 0:
            if dir==0:
                count+=0.5
                dir = 1
        # print(count)

        cv.putText(img, str(int(count)), (50, 100), cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

        ## If we want to display the count in decimals
        # cv.putText(img, str(count), (50, 100), cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)


        # # Right Arm
        # detector.findAngle(img, 12, 14, 16)

    cv.imshow("Video", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# cap.release()
cv.destroyAllWindows()