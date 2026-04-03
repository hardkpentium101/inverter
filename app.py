from pprint import pprint as print
import time
from collections import defaultdict

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

pTime=0
cTime=0

while True:
    # 1. Load an image using OpenCV
    success, img = cap.read()
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Convert the image to MediaPipe's Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # 3. Perform detection
    detection_result = detector.detect(mp_image)
    
    h, w, _ = img.shape
    # 4. Access the results
    if detection_result.hand_landmarks:
        # pprint.pprint(detection_result.hand_landmarks)
        ids = [[8,4],[4,8]]
        coords = []
        for lid, landmarks in enumerate(detection_result.hand_landmarks):
            for id, lm in enumerate(landmarks):
                if id in (4,8) and len(landmarks) == 21:
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    cv2.circle(img, (cx, cy), 1, (255,255,255) , cv2.FILLED)

                    coords.append([cx,cy])

        
        if len(coords) == 4:
            ymin = min(coords, key=lambda x: x[1])[1]

            
            coords.sort()
            coords.append(coords[0])
            for idx, coord in enumerate(coords):
                if idx == 4:
                    continue

                c = [coord, coords[idx+1]]

                c1, c2 = sorted(c,key=lambda el: el[0]) 
                for x in range(c1[0], c2[0]):
                    x1,y1,x2,y2 = c1[0], c1[1], c2[0], c2[1]
                    slope = (y2 - y1) / (x2 - x1)
                    y = int(slope * (x - x1) + y1)

                    roi = img[ymin: y+1, x:x+1]
                    
                    if roi is not None and roi.size > 0:
                        img[ymin: y+1, x:x+1] = cv2.bitwise_not(roi)

    
    # img = cv2.bitwise_not(img)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,"",(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


