import cv2 as cv
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

#defining a function which takes first parameter as the original frame 
#and second parameter a scale to rescale the video
#here, scale is 0.75 set by default which can be manipulated
def rescaleFrame(frame, scale =0.25):
    #this method works on images, videos and live
    width = int(frame.shape[1]*scale) # Here, 1 is used for original width of the video
    height = int(frame.shape[0]*scale) # Here, '0' is used for original height of video
    
    dimensions = (width, height) # defining the dimensions using new width and height using a TUPLE
    
    return cv.resize(frame, dimensions, interpolation= cv.INTER_AREA) #returning the rescaled values

# capture = cv.VideoCapture('../videos/walking.mp4')
capture = cv.VideoCapture('../videos/dance2.mp4')
# capture = cv.VideoCapture('../videos/dance3.mp4')
# capture = cv.VideoCapture('../videos/jumping.mp4')
# capture = cv.VideoCapture('../videos/skipping.mp4')

# capture = cv.VideoCapture(0)

pTime = 0
while True:
    isTrue, frame = capture.read()
    # defining the rescaled dimensions by passing 1st parameter as the original frame and then the scale
    # if no parameter is passed in 2nd place, it will take scale by default as stated in the function 'rescaleFrame'
    frame_resized = rescaleFrame(frame)
    frame_resized_rgb = cv.cvtColor(frame_resized, cv.COLOR_BGR2RGB)
    results = pose.process(frame_resized_rgb)
    # print(results.pose_landmarks)
    if(results.pose_landmarks):
        mpDraw.draw_landmarks(frame_resized, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = frame_resized.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv.circle(frame_resized, (cx, cy), 5, (255, 0, 0), -1)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv.putText(frame_resized, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # cv.imshow('Original video',frame)
    cv.imshow('Resized Video',frame_resized)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()