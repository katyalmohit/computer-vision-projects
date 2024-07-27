import cv2 as cv
import mediapipe as mp
import time
import math

    
#defining a function which takes first parameter as the original frame 
#and second parameter a scale to rescale the video
def rescaleFrame(frame, scale =0.25):
    #this method works on images, videos and live
    width = int(frame.shape[1]*scale) # Here, 1 is used for original width of the video
    height = int(frame.shape[0]*scale) # Here, '0' is used for original height of video
    
    dimensions = (width, height) # defining the dimensions using new width and height using a TUPLE
    
    return cv.resize(frame, dimensions, interpolation= cv.INTER_AREA) #returning the rescaled values



class PoseDetector():
    def __init__(self, mode=False, upperBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upperBody = upperBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
            model_complexity=1,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):


        # defining the rescaled dimensions by passing 1st parameter as the original frame and then the scale
        # if no parameter is passed in 2nd place, it will take scale by default as stated in the function 'rescaleFrame'
        # img_resized = rescaleFrame(img)
        img_resized = img
        img_resized_rgb = cv.cvtColor(img_resized, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(img_resized_rgb)
        # print(results.pose_landmarks)
        
        if(self.results.pose_landmarks):
            if draw:
               self.mpDraw.draw_landmarks(img_resized, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img_resized

    def findPosition(self, img, draw=True):
        self.lmList = []
        if(self.results.pose_landmarks):
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 0), -1)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        # _, x1, y1 = self.lmList[p1] # We can also define it like this

        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the angle
        angle = math.degrees(math.atan2(y1-y2, x1-x2)- math.atan2(y3-y2, x3-x2))
        # print(angle)

        if angle<0:
            angle+=360


        # Draw
        if draw:
            cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv.line(img, (x3, y3), (x2, y2), (0, 255, 0), 3)

            cv.circle(img, (x1, y1), 10, (0, 0, 255), -1)
            cv.circle(img, (x1, y1), 15, (255, 0, 0), 2)
            cv.circle(img, (x2, y2), 10, (0, 0, 255), -1)
            cv.circle(img, (x2, y2), 15, (255, 0, 0), 2)
            cv.circle(img, (x3, y3), 10, (0, 0, 255), -1)
            cv.circle(img, (x3, y3), 15, (255, 0, 0), 2)

            # Display angle
            cv.putText(img, str(int(angle)), (x2-50, y2+50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        
        return angle


def main():
    # capture = cv.VideoCapture('../videos/walking.mp4')
    capture = cv.VideoCapture('../videos/dance2.mp4')
    # capture = cv.VideoCapture('../videos/dance3.mp4')
    # capture = cv.VideoCapture('../videos/jumping.mp4')
    # capture = cv.VideoCapture('../videos/skipping.mp4')

    # capture = cv.VideoCapture(0)
    detector = PoseDetector()

    pTime = 0
    while True:
        isTrue, frame = capture.read()
        img = detector.findPose(frame, draw=True)
        lmList = detector.findPosition(img, draw=False)
       
        if len(lmList) !=0  :
            print(lmList[14])

            # Track a specific joint/point
            cv.circle(img, (lmList[14][1], lmList[14][2]), 10, (255, 0, 0), -1)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # cv.imshow('Original video',frame)
        cv.imshow('Resized Video',img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()