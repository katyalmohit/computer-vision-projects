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



# cap = cv.VideoCapture('../videos/dumbbells.mp4')


detector = pm.PoseDetector()

while True:
#     # success, img = cap.read()
#     # img = rescaleFrame(img)

#     # cv.imshow("Video", img)
    img = cv.imread('../videos/dips2.jpg')
    img = rescaleFrame(img, scale = 0.5)
    img = detector.findPose(img, draw = True)
    
    lmList = detector.findPosition(img, draw=False)
    print(lmList)
    cv.imshow("Image", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# cap.release()
cv.destroyAllWindows()