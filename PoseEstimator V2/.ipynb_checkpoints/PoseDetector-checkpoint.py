import cv2
import mediapipe as mp
import time
import math

class PoseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findPose(self, img, draw=True, color=(0,255,0), thickness=2):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS, self.mpDraw.DrawingSpec(color=(0,0,255), thickness=thickness, circle_radius=2), self.mpDraw.DrawingSpec(color=color, thickness=thickness, circle_radius=2))
        return img
    
    def findPosition(self, img, draw=False, color=(0,0,255), thickness=2):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), thickness, color, cv2.FILLED)
        return self.lmList
    
    def findAngle(self, img, p1, p2, p3, draw=True, color=(0,0,255), thickness=2):
        # Get the landmarks
        try:
            x1, y1 = self.lmList[p1][1:]
            x2, y2 = self.lmList[p2][1:]
            x3, y3 = self.lmList[p3][1:]    
            # Calculate the angle
            angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
            if angle < 0:
                angle += 360
            # if angle > 180:
            #     angle -= 180
            # print(angle)
            # Draw
            if draw:
                cv2.line(img, (x1, y1), (x2, y2), (255,255,255), 3)
                cv2.line(img, (x3, y3), (x2, y2), (255,255,255), 3)
                cv2.circle(img, (x1, y1), 10, (0,0,255), cv2.FILLED)
                # cv2.circle(img, (x1, y1), 15, color, 2)
                cv2.circle(img, (x2, y2), 10, (0,0,255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (0,0,255), 2)
                cv2.circle(img, (x3, y3), 10, (0,0,255), cv2.FILLED)
                cv2.putText(img, str(int(angle)), (x2-50, y2+50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
                # cv2.circle(img, (x3, y3), 15, color, 2)
                # cv2.putText(img, str(int(angle)), (x2-50, y2+50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
            return angle
        except Exception as e:
            print(e)   
            return -1 

def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        detector.findAngle(img, 12, 14, 16)
        # cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0,255,255), cv2.FILLED)
        # print(lmList)
        cv2.imshow("Image", img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Done")
 
if __name__ == "__main__":
    main();        
        
            
                
        
        