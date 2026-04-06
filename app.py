import math
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# === MediaPipe Setup ===
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# === Capture State ===
capture_active = False
capture_start_time = 0
CAPTURE_COUNTDOWN = 5  # seconds to hold before capture
DIST_THRESHOLD = 40   # adjust based on your camera scale

# === Camera Setup ===
cap = cv2.VideoCapture(0)
pTime = 0
cTime = 0

while True:
    do_capture = False
    success, img = cap.read()
    if not success:
        break

    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    detection_result = detector.detect(mp_image)

    # === Background TURBO colormap filter ===
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    img = cv2.addWeighted(img, 0.6, color, 0.4, 0)

    h, w, _ = img.shape

    if detection_result.hand_landmarks:
        coords = []
        for landmarks in detection_result.hand_landmarks:
            for id in (4, 8):
                if len(landmarks) == 21:
                    lm = landmarks[id]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # === HUD TARGET (BLUE) ===
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), 1)
                    cv2.circle(img, (cx, cy), 20, (255, 0, 0), 1)
                    cv2.line(img, (cx - 10, cy), (cx + 10, cy), (255, 0, 0), 1)
                    cv2.line(img, (cx, cy - 10), (cx, cy + 10), (255, 0, 0), 1)
                    for r in range(10, 25, 5):
                        cv2.circle(img, (cx, cy), r, (255, 0, 0), 1)
                    coords.append([cx, cy])

        # === Pixel invert between landmarks ===
        if len(coords) == 4:
            ymin = min(coords, key=lambda x: x[1])[1]
            coords.sort()
            coords.append(coords[0])
            for idx, coord in enumerate(coords):
                if idx == 4:
                    continue
                c = [coord, coords[idx + 1]]
                c1, c2 = sorted(c, key=lambda el: el[0])
                for x in range(c1[0], c2[0]):
                    x1, y1, x2, y2 = c1[0], c1[1], c2[0], c2[1]
                    slope = (y2 - y1) / (x2 - x1)
                    y = int(slope * (x - x1) + y1)
                    roi = img[ymin:y + 1, x:x + 1]
                    if roi is not None and roi.size > 0:
                        img[ymin:y + 1, x:x + 1] = cv2.bitwise_not(roi)

        # === Line between index fingertips + capture gesture ===
        if len(coords) >= 2:
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

            if len(detection_result.hand_landmarks) >= 2:
                ring1 = detection_result.hand_landmarks[0][16]
                ring2 = detection_result.hand_landmarks[1][16]
                rx1, ry1 = int(ring1.x * w), int(ring1.y * h)
                rx2, ry2 = int(ring2.x * w), int(ring2.y * h)
                ring_dist = math.hypot(rx2 - rx1, ry2 - ry1)

                if ring_dist < DIST_THRESHOLD and not capture_active:
                    capture_active = True
                    capture_start_time = time.time()

                if capture_active:
                    elapsed = time.time() - capture_start_time
                    remaining = max(0, CAPTURE_COUNTDOWN - int(elapsed))
                    cv2.putText(img, f"CAPTURE: {remaining}",
                                (w // 2 - 100, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    progress = elapsed / CAPTURE_COUNTDOWN
                    bar_width = int(200 * min(progress, 1.0))
                    cv2.rectangle(img, (w // 2 - 100, 65),
                                  (w // 2 + 100, 80), (50, 50, 50), -1)
                    cv2.rectangle(img, (w // 2 - 100, 65),
                                  (w // 2 - 100 + bar_width, 80), (0, 255, 255), -1)

                    if elapsed >= CAPTURE_COUNTDOWN:
                        capture_active = False
                        do_capture = True

    # === FPS Counter ===
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # === Capture after all rendering ===
    if do_capture:
        capture_filename = f"capture_{int(time.time())}.png"
        cv2.imwrite(capture_filename, img)
        print(f"Image captured: {capture_filename}")
        cv2.putText(img, "CAPTURED!",
                    (w // 2 - 80, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Hand Landmark Inverter", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


