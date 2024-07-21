import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils

# We can also specify the Detection Confidence
faceDetection = mpFaceDetection.FaceDetection(0.5)


def rescaleFrame(frame, scale =1):
    #this method works on images, videos and live
    width = int(frame.shape[1]*scale) # Here, 1 is used for original width of the video
    height = int(frame.shape[0]*scale) # Here, '0' is used for original height of video
    
    dimensions = (width, height) # defining the dimensions using new width and height using a TUPLE
    
    return cv.resize(frame, dimensions, interpolation= cv.INTER_AREA) #returning the rescaled values

pTime = 0
while True:
    success, img = cap.read()
    img_resized = rescaleFrame(img)
    img_res_rgb = cv.cvtColor(img_resized, cv.COLOR_BGR2RGB)
    results = faceDetection.process(img_res_rgb)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img_resized, detection)
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)

            ## if we want to draw with other method
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img_resized.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
            cv.rectangle(img_resized, bbox, (255, 0, 255), 2)
            cv.putText(img_resized, f"{int(detection.score[0] * 100) }%", (bbox[0], bbox[1]-20), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img_resized, f"FPS: {int(fps)}", (10, 35), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
    cv.imshow("Video", img_resized)

    if cv.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()