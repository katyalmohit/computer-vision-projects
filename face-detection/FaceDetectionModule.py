import cv2 as cv
import mediapipe as mp
import time


def rescaleFrame(frame, scale =1):
    #this method works on images, videos and live
    width = int(frame.shape[1]*scale) # Here, 1 is used for original width of the video
    height = int(frame.shape[0]*scale) # Here, '0' is used for original height of video
    
    dimensions = (width, height) # defining the dimensions using new width and height using a TUPLE
    
    return cv.resize(frame, dimensions, interpolation= cv.INTER_AREA) #returning the rescaled values



class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils 
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):

        img_res_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(img_res_rgb)
        # print(self.results)
        bboxs = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                        int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    self.fancyDraw(img, bbox, l=30, t=5, rt=1)
                
                cv.putText(img, f"{int(detection.score[0] * 100) }%", (bbox[0], bbox[1]-20), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        return img, bboxs
   
    def fancyDraw(self, img, bbox, l=30, t=10, rt=1):
        # In the parameters we have specified the image, the bounding box, length, thickness, rectangle thickness
            x, y, w, h = bbox
            x1, y1 = x + w, y+h

            cv.rectangle(img, bbox, (255, 0, 255), rt)

            # Top left x, y
            cv.line(img, (x, y), (x+l, y), (255, 0, 255), t)
            cv.line(img, (x, y), (x , y+l), (255, 0, 255), t)

            # Top right x, y
            cv.line(img, (x1, y), (x1-l, y), (255, 0, 255), t)
            cv.line(img, (x1, y), (x1 , y+l), (255, 0, 255), t)

            # Bottom left x, y
            cv.line(img, (x, y1), (x+l, y1), (255, 0, 255), t)
            cv.line(img, (x, y1), (x , y1-l), (255, 0, 255), t)

            # Bottom right x, y
            cv.line(img, (x1, y1), (x1-l, y1), (255, 0, 255), t)
            cv.line(img, (x1, y1), (x1 , y1-l), (255, 0, 255), t)

            return img


def main():

    cap = cv.VideoCapture(0)
    pTime = 0
    detector = FaceDetector(0.4)

    while True:
        success, img = cap.read()
        img_resized = rescaleFrame(img)
        img_resized, bboxs = detector.findFaces(img_resized, draw=True)
        print(bboxs)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv.putText(img_resized, f"FPS: {int(fps)}", (10, 35), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
        cv.imshow("Video", img_resized)

        if cv.waitKey(20) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()