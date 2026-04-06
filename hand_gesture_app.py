import math
import random
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# === MediaPipe Setup (Tasks API - same as app.py) ===
hand_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
hand_detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=hand_options,
        num_hands=2
    )
)

face_options = python.BaseOptions(model_asset_path='face_landmarker.task')
face_detector = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=face_options,
        num_faces=1
    )
)

# === Constants ===
DIST_THRESHOLD = 40  # pixels between index fingertips
CHARGE_TIME = 2      # seconds to hold before activation

# === 3D Shape Projection ===
# 8 cube vertices in 3D (normalized -1 to 1)
CUBE_VERTS_3D = [
    (-1, -1, -1), ( 1, -1, -1), ( 1,  1, -1), (-1,  1, -1),
    (-1, -1,  1), ( 1, -1,  1), ( 1,  1,  1), (-1,  1,  1),
]

# 6 cube faces
CUBE_FACES = [
    [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
    [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5],
]

# Octahedron (8 triangular faces)
OCTA_VERTS = [
    ( 1,  0,  0), (-1,  0,  0),
    ( 0,  1,  0), ( 0, -1,  0),
    ( 0,  0,  1), ( 0,  0, -1),
]
OCTA_FACES = [
    [0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],
    [1, 4, 2], [1, 3, 4], [1, 5, 3], [1, 2, 5],
]

# Icosphere (subdivided icosahedron approximates sphere, 20 faces)
SPH_PHI = (1 + 5**0.5) / 2
SPH_SCALE = 1.0 / SPH_PHI
SPH_VERTS = [
    ( 0,  SPH_SCALE,  1), ( 0,  SPH_SCALE, -1),
    ( 0, -SPH_SCALE,  1), ( 0, -SPH_SCALE, -1),
    ( 1,  SPH_SCALE,  0), (-1,  SPH_SCALE,  0),
    ( 1, -SPH_SCALE,  0), (-1, -SPH_SCALE,  0),
    ( SPH_SCALE,  0,  1), (-SPH_SCALE,  0,  1),
    ( SPH_SCALE,  0, -1), (-SPH_SCALE,  0, -1),
]
# Normalize sphere verts to radius ~1
_smax = max(math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) for v in SPH_VERTS)
SPH_VERTS = [(v[0]/_smax, v[1]/_smax, v[2]/_smax) for v in SPH_VERTS]

SPH_FACES = [
    [0, 8, 4], [0, 4, 5], [0, 5, 9], [0, 9, 8],
    [8, 9, 2], [2, 9, 7], [2, 7, 6], [2, 6, 8],
    [4, 8, 6], [6,10, 4], [4,10,11], [4,11, 5],
    [5,11, 3], [3,11,10], [3,10, 7], [3, 7, 9],
    [1, 5,11], [1,10, 6], [1,11, 5], [1, 6,10],
]

# Shape definitions: Cube, Octahedron, Sphere
SHAPES = [
    {"name": "Cube",         "verts": CUBE_VERTS_3D,   "faces": CUBE_FACES},
    {"name": "Octahedron",   "verts": OCTA_VERTS,     "faces": OCTA_FACES},
    {"name": "Sphere",       "verts": SPH_VERTS,      "faces": SPH_FACES},
]

# Vivid per-shape colors (R, G, B) — full saturation, bright
SHAPE_COLORS = [
    (0, 50, 255),     # pure red - cube
    (255, 180, 0),    # bright cyan-teal - octahedron
    (200, 255, 50),   # vivid magenta-gold - sphere
]


def is_fist(hand_lm):
    """Check if a hand is making a fist (3+ fingertips below MCP joints)"""
    tip_ids = [8, 12, 16, 20]
    mcp_ids = [5, 9, 13, 17]
    folded = sum(1 for t, m in zip(tip_ids, mcp_ids) if hand_lm[t].y > hand_lm[m].y)
    return folded >= 3


def project_3d_to_2d(point_3d, center, size, rot_y, rot_x):
    """Project a 3D point to 2D with Y and X rotation"""
    x, y, z = point_3d

    # Rotate around Y axis (yaw)
    rad_y = math.radians(rot_y)
    cos_y, sin_y = math.cos(rad_y), math.sin(rad_y)
    x_rot = x * cos_y - z * sin_y
    z_rot = x * sin_y + z * cos_y

    # Rotate around X axis (tilt)
    rad_x = math.radians(rot_x)
    cos_x, sin_x = math.cos(rad_x), math.sin(rad_x)
    y_final = y * cos_x - z_rot * sin_x
    z_final = y * sin_x + z_rot * cos_x

    # Perspective projection
    scale = size / (3 + z_final)
    x_2d = center[0] + int(x_rot * scale)
    y_2d = center[1] + int(y_final * scale)

    return (x_2d, y_2d)


# === Eye Animation State ===
class EyeAnimation:
    def __init__(self):
        self.state = "idle"
        self.charge_progress = 0.0
        self.rotation_angle = 0.0
        self.intensity = 0.0
        self.charge_start = 0

    def update(self, fingers_touching, frame_time):
        if self.state == "idle":
            if fingers_touching:
                self.state = "charging"
                self.charge_start = frame_time
                self.charge_progress = 0.0

        elif self.state == "charging":
            elapsed = frame_time - self.charge_start
            self.charge_progress = min(1.0, elapsed / CHARGE_TIME)
            if not fingers_touching:
                self.state = "idle"
                self.charge_progress = 0.0
            elif self.charge_progress >= 1.0:
                self.state = "active"
                self.intensity = 1.0  # MAX intensity

        elif self.state == "active":
            self.rotation_angle += 12  # faster rotation
            # No fading — stays active forever once triggered


def draw_sharingan(center, radius, rotation, intensity, frame):
    """Draw Sharingan eye effect - RIGHT EYE ONLY with MAXIMUM intensity"""
    cx, cy = center
    overlay = frame.copy()

    # Massive red glow
    glow_radius = int(radius * 3.0 * intensity)
    for r in range(glow_radius, 0, -3):
        alpha = (r / glow_radius) * 0.6 * intensity
        glow_overlay = frame.copy()
        cv2.circle(glow_overlay, (cx, cy), r, (0, 0, 255), -1)
        cv2.addWeighted(glow_overlay, alpha, frame, 1 - alpha, 0, frame)

    # Outer pulsing glow
    pulse = 1.0 + 0.3 * math.sin(math.radians(rotation * 3))
    pulse_radius = int(glow_radius * pulse)
    pulse_overlay = frame.copy()
    cv2.circle(pulse_overlay, (cx, cy), pulse_radius, (0, 0, 255), -1)
    cv2.addWeighted(pulse_overlay, 0.15 * intensity, frame, 1 - 0.15 * intensity, 0, frame)

    # Dark pupil - much larger
    pupil_radius = int(radius * 0.7 * intensity)
    cv2.circle(overlay, (cx, cy), pupil_radius, (0, 0, 0), -1)

    # Outer ring - very thick
    ring_radius = int(radius * 1.0 * intensity)
    cv2.circle(overlay, (cx, cy), ring_radius, (0, 0, 0), 5)

    # Inner ring - thick
    inner_ring = int(radius * 0.7 * intensity)
    cv2.circle(overlay, (cx, cy), inner_ring, (0, 0, 0), 3)

    # Core ring
    core_ring = int(radius * 0.4 * intensity)
    cv2.circle(overlay, (cx, cy), core_ring, (0, 0, 0), 2)

    # 3 tomoe - much larger and more pronounced
    if intensity > 0.2:
        for i in range(3):
            angle = math.radians(rotation + i * 120)
            tomoe_x = int(cx + radius * 0.75 * math.cos(angle))
            tomoe_y = int(cy + radius * 0.75 * math.sin(angle))

            tomoe_size = int(radius * 0.3 * intensity)
            cv2.circle(overlay, (tomoe_x, tomoe_y), tomoe_size, (0, 0, 0), -1)

            # Tomoe tail - much longer
            tail_angle = angle + math.radians(30)
            tail_x = int(tomoe_x + radius * 0.4 * math.cos(tail_angle))
            tail_y = int(tomoe_y + radius * 0.4 * math.sin(tail_angle))
            cv2.line(overlay, (tomoe_x, tomoe_y), (tail_x, tail_y), (0, 0, 0), 4)

            # Tomoe glow
            if intensity > 0.5:
                cv2.circle(overlay, (tomoe_x, tomoe_y), tomoe_size + 3, (50, 0, 0), 2)

    # Blend with maximum alpha
    alpha = 0.95 * intensity
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_cube_on_hands(frame, hand_landmarks, h, w, intensity, proximity_factor, cube_size, rot_y, rot_x,
                       shape_idx=0, morph_prog=1.0, is_exploded=False, particle_list=None):
    """Draw 3D shape centered on screen, purple surface glow, fingertips to fingertips"""

    has_hands = len(hand_landmarks) >= 2
    shape = SHAPES[shape_idx]
    verts_3d = shape["verts"]
    faces = shape["faces"]

    # Collect all fingertip positions (thumb, index, middle, ring, pinky from each hand)
    all_finger_ids = [4, 8, 12, 16, 20]
    tips = []
    if has_hands:
        for hand_idx in range(2):
            landmarks = hand_landmarks[hand_idx]
            for fid in all_finger_ids:
                x = int(landmarks[fid].x * w)
                y = int(landmarks[fid].y * h)
                tips.append((x, y))

    # Cube center = fixed center of screen
    cube_center = (w // 2, h // 2)

    # Size grows continuously with proximity, never shrinks (reset only on thumb+index touch)
    if proximity_factor > 0.1:
        cube_size = max(cube_size, int(min(h, w) * 0.1))
        growth = proximity_factor * 15
        cube_size += growth
    cube_size = min(cube_size, int(min(h, w) * 0.8))

    # Project 3D vertices to 2D
    verts_2d = [project_3d_to_2d(v, cube_center, cube_size, rot_y, rot_x) for v in verts_3d]

    # Sort faces by average Z for painter's algorithm (back to front)
    face_z_values = []
    for face in faces:
        avg_z = sum(verts_3d[i][2] for i in face) / len(face)
        face_z_values.append((avg_z, face))
    face_z_values.sort(key=lambda f: f[0])

    # Shape color
    color = SHAPE_COLORS[shape_idx]

    # Draw filled faces — OUTER ONLY (no internal geometry)
    for _, face in face_z_values:
        pts = np.array([verts_2d[i] for i in face], dtype=np.int32)

        # Deep base fill
        base_intensity = 0.4 + proximity_factor * 0.6
        cv2.fillPoly(frame, [pts], (
            int(color[0] * 0.4 * base_intensity),
            int(color[1] * 0.4 * base_intensity),
            int(color[2] * 0.4 * base_intensity)
        ))

        # Layered glow
        glow_mult = 1.0 if morph_prog >= 1.0 else morph_prog
        for layer in range(6, 0, -1):
            alpha = (layer / 6) * 0.22 * max(proximity_factor, glow_mult * 0.6, 0.3)
            glow_overlay = frame.copy()
            cv2.fillPoly(glow_overlay, [pts], (
                int(color[0] * max(proximity_factor, glow_mult * 0.6, 0.3)),
                int(color[1] * max(proximity_factor, glow_mult * 0.6, 0.3)),
                int(color[2] * max(proximity_factor, glow_mult * 0.6, 0.3))
            ))
            cv2.addWeighted(glow_overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw edges
    edge_alpha = int(200 * max(proximity_factor, 0.3))
    edge_set = set()
    for face in faces:
        for i in range(len(face)):
            a, b = sorted((face[i], face[(i + 1) % len(face)]))
            edge_set.add((a, b))

    for a, b in edge_set:
        p1, p2 = verts_2d[a], verts_2d[b]
        cv2.line(frame, p1, p2, (edge_alpha, int(edge_alpha * 0.5), edge_alpha), 1)

    # Draw particles (explode/implode)
    if particle_list:
        p_color = SHAPE_COLORS[shape_idx]
        for p in particle_list:
            if p["life"] <= 0:
                continue
            try:
                px = int(p["x"])
                py = int(p["y"])
            except (ValueError, TypeError):
                continue
            alpha = p["life"]
            glow_color = (
                int(p_color[0] * alpha),
                int(p_color[1] * alpha),
                int(p_color[2] * alpha)
            )
            center = (px, py)
            for thickness in range(p["size"], 0, -1):
                g_alpha = (thickness / p["size"]) * 0.3 * alpha
                g_overlay = frame.copy()
                cv2.circle(g_overlay, center, thickness, glow_color, -1)
                cv2.addWeighted(g_overlay, g_alpha, frame, 1 - g_alpha, 0, frame)
            # Core
            cv2.circle(frame, center, max(1, p["size"] // 3), (255, 255, 255), -1)

    # Connect fingertips to fingertips (hand 0 to hand 1, same finger type)
    if has_hands:
        for i, fid in enumerate(all_finger_ids):
            tip0 = (
                int(hand_landmarks[0][fid].x * w),
                int(hand_landmarks[0][fid].y * h)
            )
            tip1 = (
                int(hand_landmarks[1][fid].x * w),
                int(hand_landmarks[1][fid].y * h)
            )
            # Light blue connection line with lower intensity
            line_color = (180, 200, 255)  # BGR: light blue
            cv2.line(frame, tip0, tip1, line_color, 1)

    return cube_size


cap = cv2.VideoCapture(0)

pTime = 0
animation = EyeAnimation()
last_eye_radius = 60
last_right_eye = (0, 0)
persistent_cube_size = 0  # grows with proximity, resets on thumb+index touch

# Rotation state
cube_rot_y = 0.0  # Y-axis rotation (yaw)
cube_rot_x = 25.0  # X-axis tilt (degrees, base offset for visibility)
prev_hand_centers = None  # for velocity tracking
rot_velocity_y = 0.0  # accumulated rotational momentum

# Shape morph state
current_shape_idx = 0  # 0=Cube, 1=Octahedron, 2=Icosahedron, 3=Dodecahedron
morph_progress = 1.0  # 1.0 = fully morphed, 0.0 = transitioning
prev_both_fist = False  # detect clench→open
shape_morph_cooldown = 0  # frames before next morph allowed

# Explode/Implode state
particles = []  # list of particle dicts
exploded = False  # shape is currently exploded
implode_started = False  # particles returning to center

# Explode detection: track fist clench → distance increase over time
explode_tracking = None  # None or {"start_time", "start_dist"}
EXPLODE_WINDOW = 1.5  # seconds to detect the pull-apart
EXPLODE_DIST_RATIO = 1.6  # distance must increase by 60% to trigger

# Capture timer state (ring fingertips - ID 16 - NOT preassigned)
capture_active = False
capture_start_time = 0
CAPTURE_COUNTDOWN = 5  # seconds to hold before capture

while True:
    do_capture = False  # flag set by capture timer, checked after rendering
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # === Face Detection ===
    face_detected = False

    face_result = face_detector.detect(mp_img)
    if face_result.face_landmarks:
        landmarks = face_result.face_landmarks[0]
        face_detected = True

        # RIGHT eye only
        right_ids = [362, 263, 387, 386, 385, 373, 374, 380]
        right_eye = [(landmarks[i].x * w, landmarks[i].y * h) for i in right_ids]

        last_right_eye = (int(sum(p[0] for p in right_eye) / len(right_eye)),
                          int(sum(p[1] for p in right_eye) / len(right_eye)))

        # Face size as distance proxy
        forehead = landmarks[10]
        chin = landmarks[152]
        face_h = math.hypot((chin.x - forehead.x) * w, (chin.y - forehead.y) * h)
        face_w = math.hypot((landmarks[234].x - landmarks[454].x) * w,
                            (landmarks[234].y - landmarks[454].y) * h)
        face_diagonal = math.hypot(face_w, face_h)

        last_eye_radius = int(face_diagonal * 0.1) // 3

    right_eye_center = last_right_eye
    eye_radius = last_eye_radius

    # === Hand Detection ===
    hand_result = hand_detector.detect(mp_img)
    fingers_touching = False
    proximity_factor = 0.0
    size_reset = False

    if hand_result.hand_landmarks and len(hand_result.hand_landmarks) >= 2:
        # NO glow on hands (removed)

        # Check distance between index fingertips
        tip1 = hand_result.hand_landmarks[0][8]
        tip2 = hand_result.hand_landmarks[1][8]
        x1, y1 = int(tip1.x * w), int(tip1.y * h)
        x2, y2 = int(tip2.x * w), int(tip2.y * h)
        dist = math.hypot(x2 - x1, y2 - y1)

        # Proximity factor: 0 = far, 1 = very close (wider reaction range)
        proximity_factor = max(0.0, 1.0 - (dist / (DIST_THRESHOLD * 5)))

        # === CAPTURE GESTURE: Ring fingertips (ID 16) touching ===
        ring_tip1 = hand_result.hand_landmarks[0][16]
        ring_tip2 = hand_result.hand_landmarks[1][16]
        ring_x1 = int(ring_tip1.x * w)
        ring_y1 = int(ring_tip1.y * h)
        ring_x2 = int(ring_tip2.x * w)
        ring_y2 = int(ring_tip2.y * h)
        ring_dist = math.hypot(ring_x2 - ring_x1, ring_y2 - ring_y1)

        if ring_dist < DIST_THRESHOLD and not capture_active:
            capture_active = True
            capture_start_time = time.time()

        if capture_active:
            elapsed = time.time() - capture_start_time
            remaining = max(0, CAPTURE_COUNTDOWN - int(elapsed))

            # Draw countdown timer
            cv2.putText(img, f"CAPTURE: {remaining}",
                        (w // 2 - 100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

            # Progress bar
            progress = elapsed / CAPTURE_COUNTDOWN
            bar_width = int(200 * min(progress, 1.0))
            cv2.rectangle(img, (w // 2 - 100, 65),
                          (w // 2 + 100, 80), (50, 50, 50), -1)
            cv2.rectangle(img, (w // 2 - 100, 65),
                          (w // 2 - 100 + bar_width, 80), (0, 255, 255), -1)

            # Draw ring finger indicator
            cv2.circle(img, ((ring_x1 + ring_x2) // 2, (ring_y1 + ring_y2) // 2),
                       15, (0, 255, 255), 2)

            if elapsed >= CAPTURE_COUNTDOWN:
                capture_active = False

                # Flag to trigger capture after all rendering
                do_capture = True

        # === SHAPE MORPH: Fist clench → open quickly ===
        both_fist = is_fist(hand_result.hand_landmarks[0]) or is_fist(hand_result.hand_landmarks[1])
        if prev_both_fist and not both_fist and shape_morph_cooldown <= 0:
            current_shape_idx = (current_shape_idx + 1) % len(SHAPES)
            morph_progress = 0.0
            shape_morph_cooldown = 30
        prev_both_fist = both_fist
        if shape_morph_cooldown > 0:
            shape_morph_cooldown -= 1

        # === EXPLODE/IMPLODE ===
        both_hands_fist = (is_fist(hand_result.hand_landmarks[0]) and
                          is_fist(hand_result.hand_landmarks[1]))
        current_dist = math.hypot(x2 - x1, y2 - y1)

        if not exploded:
            if both_hands_fist and explode_tracking is None:
                # Start tracking: both fists clenched
                explode_tracking = {
                    "start_time": time.time(),
                    "start_dist": current_dist,
                }
            elif both_hands_fist and explode_tracking is not None:
                # Still fists — check if distance increased enough within window
                elapsed = time.time() - explode_tracking["start_time"]
                if elapsed <= EXPLODE_WINDOW:
                    if explode_tracking["start_dist"] > 5:
                        ratio = current_dist / explode_tracking["start_dist"]
                        if ratio >= EXPLODE_DIST_RATIO:
                            # EXPLODE
                            exploded = True
                            implode_started = False
                            explode_tracking = None
                            shape = SHAPES[current_shape_idx]
                            spawn_center = (w // 2, h // 2)
                            for vert in shape["verts"]:
                                p2d = project_3d_to_2d(vert, spawn_center, persistent_cube_size, cube_rot_y, cube_rot_x)
                                for _ in range(8):
                                    angle = random.uniform(0, 2 * math.pi)
                                    speed = random.uniform(3, 12)
                                    particles.append({
                                        "x": float(p2d[0]),
                                        "y": float(p2d[1]),
                                        "vx": math.cos(angle) * speed,
                                        "vy": math.sin(angle) * speed - 3,
                                        "life": 1.0,
                                        "decay": random.uniform(0.006, 0.015),
                                        "size": random.randint(3, 10),
                                    })
                else:
                    # Window expired, reset tracking
                    explode_tracking = None
            else:
                # Not both fists — reset tracking
                explode_tracking = None
        else:
            # Implode trigger: open both hands while exploded
            both_open = (not is_fist(hand_result.hand_landmarks[0]) and
                        not is_fist(hand_result.hand_landmarks[1]))
            if both_open and not implode_started and len(particles) > 5:
                implode_started = True

        if dist < DIST_THRESHOLD:
            fingers_touching = True
            tx, ty = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(img, (tx, ty), 15, (0, 255, 255), 2)

        # Size reset: thumb-to-thumb + either hand fist
        thumb0_pt = hand_result.hand_landmarks[0][4]
        thumb1_pt = hand_result.hand_landmarks[1][4]
        thumb_to_thumb = math.hypot((thumb0_pt.x - thumb1_pt.x) * w, (thumb0_pt.y - thumb1_pt.y) * h)
        size_reset = thumb_to_thumb < 25 and (is_fist(hand_result.hand_landmarks[0]) or
                                               is_fist(hand_result.hand_landmarks[1]))

        # === ROTATION INPUTS ===
        hand_centers = []
        for hm in hand_result.hand_landmarks:
            all_pts = [(int(hm[i].x * w), int(hm[i].y * h)) for i in range(21)]
            cx = sum(p[0] for p in all_pts) / len(all_pts)
            cy = sum(p[1] for p in all_pts) / len(all_pts)
            hand_centers.append((cx, cy))

        fist_hand = None
        open_hand = None
        fist0 = is_fist(hand_result.hand_landmarks[0])
        fist1 = is_fist(hand_result.hand_landmarks[1])

        if fist0 and not fist1:
            fist_hand, open_hand = 0, 1
        elif fist1 and not fist0:
            fist_hand, open_hand = 1, 0

        if fist_hand is not None and open_hand is not None:
            open_lm = hand_result.hand_landmarks[open_hand]
            open_pts = [(int(open_lm[i].x * w), int(open_lm[i].y * h)) for i in range(21)]
            oh_cx = sum(p[0] for p in open_pts) / len(open_pts)
            oh_cy = sum(p[1] for p in open_pts) / len(open_pts)

            y_norm = oh_cy / h
            target_tilt = 5.0 + y_norm * 50.0
            cube_rot_x += (target_tilt - cube_rot_x) * 0.08

            fist_lm = hand_result.hand_landmarks[fist_hand]
            fist_pts = [(int(fist_lm[i].x * w), int(fist_lm[i].y * h)) for i in range(21)]
            fist_cx = sum(p[0] for p in fist_pts) / len(fist_pts)

            x_offset = (oh_cx - fist_cx) / w
            yaw_force = x_offset * 8.0
        else:
            dist_ratio = 1.0 - proximity_factor
            yaw_force = 1.0 + (1.0 - dist_ratio) * 5.0

            avg_hand_y = (hand_centers[0][1] + hand_centers[1][1]) / 2
            hand_y_norm = avg_hand_y / h
            target_tilt = 25.0 + (hand_y_norm - 0.5) * 40.0
            cube_rot_x += (target_tilt - cube_rot_x) * 0.05

        if fist_hand is None and prev_hand_centers and len(hand_centers) == 2:
            vel_sum = 0.0
            for i, hc in enumerate(hand_centers):
                dx = hc[0] - prev_hand_centers[i][0]
                dy = hc[1] - prev_hand_centers[i][1]
                vel_sum += math.hypot(dx, dy)
            rot_velocity_y += vel_sum * 0.003

        if fist_hand is not None:
            cube_rot_y += yaw_force
        else:
            rot_velocity_y = rot_velocity_y * 0.92 + yaw_force * 0.08
            cube_rot_y += rot_velocity_y

        if fingers_touching:
            rot_velocity_y *= 0.9

        prev_hand_centers = list(hand_centers)

    elif animation.state == "active":
        cube_rot_y += 0.5
        rot_velocity_y *= 0.95
    else:
        cube_rot_y += 0.5
        rot_velocity_y *= 0.95

    # === PARTICLE UPDATE (always, even without hands) ===
    if exploded and len(particles) > 0:
        center = (w // 2, h // 2)
        if not implode_started:
            for p in particles:
                p["x"] += p["vx"]
                p["y"] += p["vy"]
                p["vy"] += 0.15
                p["life"] -= p["decay"]
        else:
            for p in particles:
                dx = center[0] - p["x"]
                dy = center[1] - p["y"]
                dist_to_center = math.hypot(dx, dy)
                if dist_to_center > 1:
                    p["vx"] = dx / dist_to_center * min(12, 15 - dist_to_center * 0.1)
                    p["vy"] = dy / dist_to_center * min(12, 15 - dist_to_center * 0.1)
                p["x"] += p["vx"]
                p["y"] += p["vy"]
                p["life"] -= p["decay"] * 0.5
                if dist_to_center < 20:
                    p["life"] = 0
        particles = [p for p in particles if p["life"] > 0]
        if len(particles) == 0 and implode_started:
            exploded = False
            implode_started = False

    # === Animation ===
    frame_time = time.time()
    animation.update(fingers_touching, frame_time)

    # Draw RIGHT EYE ONLY
    if face_detected and animation.intensity > 0.01:
        draw_sharingan(right_eye_center, eye_radius,
                       animation.rotation_angle, animation.intensity, img)

    # Morph progress smooth transition
    if morph_progress < 1.0:
        morph_progress = min(1.0, morph_progress + 0.03)

    # Draw 3D shape when animation is active — remains even without hands
    if animation.state == "active":
        # Reset cube size when thumbs touch (inter-hand)
        if size_reset:
            persistent_cube_size = 0

        # Shape always renders when active; hands are optional
        cube_hands = hand_result.hand_landmarks if hand_result.hand_landmarks and len(hand_result.hand_landmarks) >= 2 else []
        persistent_cube_size = draw_cube_on_hands(
            img, cube_hands, h, w,
            animation.intensity, proximity_factor, persistent_cube_size,
            cube_rot_y, cube_rot_x,
            shape_idx=current_shape_idx, morph_prog=morph_progress,
            is_exploded=exploded, particle_list=particles
        )

        # Show current shape name
        cv2.putText(img, SHAPES[current_shape_idx]["name"],
                    (w // 2 - 60, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # === HUD ===
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    state_colors = {
        "idle": (255, 255, 255),
        "charging": (0, 255, 255),
        "active": (0, 0, 255),
        "fading": (128, 128, 128)
    }

    if animation.state == "charging":
        bar_width = int(200 * animation.charge_progress)
        cv2.rectangle(img, (10, 80), (210, 95), (100, 100, 100), -1)
        cv2.rectangle(img, (10, 80), (10 + bar_width, 95), (0, 255, 255), -1)
        cv2.putText(img, "Charging...", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # === Background TURBO colormap filter ===
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    img = cv2.addWeighted(img, 0.6, color, 0.4, 0)

    # === Capture after all rendering ===
    if do_capture:
        capture_filename = f"capture_{int(time.time())}.png"
        cv2.imwrite(capture_filename, img)
        print(f"Image captured: {capture_filename}")
        # Flash effect
        cv2.putText(img, "CAPTURED!",
                    (w // 2 - 80, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Hand Gesture Interface", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
