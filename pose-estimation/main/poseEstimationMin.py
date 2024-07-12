import cv2 as cv
import mediapipe as mp



#defining a function which takes first parameter as the original frame 
#and second parameter a scale to rescale the video
#here, scale is 0.75 set by default which can be manipulated
def rescaleFrame(frame, scale =0.5):
    #this method works on images, videos and live
    width = int(frame.shape[1]*scale) # Here, 1 is used for original width of the video
    height = int(frame.shape[0]*scale) # Here, '0' is used for original height of video
    
    dimensions = (width, height) # defining the dimensions using new width and height using a TUPLE
    
    return cv.resize(frame, dimensions, interpolation= cv.INTER_AREA) #returning the rescaled values

capture = cv.VideoCapture('../videos/walking.mp4')
# capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()
    # defining the rescaled dimensions by passing 1st parameter as the original frame and then the scale
    # if no parameter is passed in 2nd place, it will take 0.5 by default as stated in the function 'rescaleFrame'
    frame_resized = rescaleFrame(frame)

    
    # cv.imshow('Original video',frame)
    cv.imshow('Resized Video',frame_resized)
    
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()