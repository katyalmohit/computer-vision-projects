import cv2 as cv
import mediapipe as mp
import time


class FaceMeshDetector():
    def __init__(self, staticMode = False, maxFaces = 2, minDetectionCon = 0.5, minTrackCon = 0.5, thickness=1, circle_radius=1, color=(0,255, 0)):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.thickness = thickness
        self.circle_radius = circle_radius
        self.color = color

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
                                static_image_mode = self.staticMode,
                                max_num_faces =self.maxFaces,
                                min_detection_confidence = self.minDetectionCon,
                                min_tracking_confidence= self.minTrackCon
                            )
        
        self.drawSpec = self.mpDraw.DrawingSpec(
                                thickness = thickness, 
                                circle_radius = circle_radius, 
                                color = color
                            )


    def findFaceMesh(self, img, draw=True):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(img_rgb)

        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)
                    # self.mpDraw.draw_landmarks(img, faceLms) # If we don't want to display connections

                    
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    
                    ##Display id No. on the respective landmark
                    # cv.putText(img, str(id), (x, y), cv.FONT_HERSHEY_PLAIN, 0.4, (0, 255, 0), 1)
                    # print(id, x, y)
                    face.append([x, y])
                faces.append(face)
        return img, faces
   

def main():
        
    cap = cv.VideoCapture(0)
    pTime = 0

    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()

        img, faces = detector.findFaceMesh(img, draw= True)
        if len(faces)!=0:
            print(len(faces[0]))
            # print(faces[0])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv.putText(img, f"FPS: {int(fps)}", (10, 35), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv.imshow("Video", img)

        if cv.waitKey(20) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()