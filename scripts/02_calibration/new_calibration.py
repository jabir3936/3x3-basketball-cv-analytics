import cv2
import json
import math
import numpy as np

VIDEO_PATH = "media/footage/court_footage.mp4"
FRAME_ID = 100

# ---------------------------------------------------------
# LANDMARK PAIRS: pixel -> official meters
# (Adjust the pixel values if the overlay is off)
# ---------------------------------------------------------
PIXEL_POINTS = [
    (585, 520),    # key left  @ baseline
    (1195, 470),   # key right @ baseline
    (620, 700),    # key left  @ free-throw line
    (1250, 640),   # key right @ free-throw line
    (375, 570),    # arc straight left  @ baseline
    (1440, 540),   # arc straight right @ baseline
]

OFFICIAL_POINTS = [
    (-2.45, 0.0),
    ( 2.45, 0.0),
    (-2.45, 5.80),
    ( 2.45, 5.80),
    (-6.60, 0.0),
    ( 6.60, 0.0),
]

H2, _ = cv2.findHomography(
    np.float32(PIXEL_POINTS),
    np.float32(OFFICIAL_POINTS)
)

with open("data/calibration/calibration_official.json", "w") as f:
    json.dump({"homography_matrix": H2.tolist()}, f, indent=2)

H2_inv = np.linalg.inv(H2)

def off_to_pixel(p):
    v = np.array([p[0], p[1], 1.0], dtype=np.float32)
    t = H2_inv @ v
    return (int(round(t[0] / t[2])), int(round(t[1] / t[2])))

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_ID - 1)
ret, frame = cap.read()
cap.release()

# Boundary (yellow)
for a, b in [((-7.5,0),(7.5,0)), ((7.5,0),(7.5,11)),
             ((7.5,11),(-7.5,11)), ((-7.5,11),(-7.5,0))]:
    cv2.line(frame, off_to_pixel(a), off_to_pixel(b), (255,255,0), 2)

# Key (cyan)
for a, b in [((-2.45,0),(2.45,0)), ((2.45,0),(2.45,5.8)),
             ((2.45,5.8),(-2.45,5.8)), ((-2.45,5.8),(-2.45,0))]:
    cv2.line(frame, off_to_pixel(a), off_to_pixel(b), (0,255,255), 2)

# Arc straights + arc (green)
cv2.line(frame, off_to_pixel((-6.6,0)), off_to_pixel((-6.6,2.99)), (0,255,0), 2)
cv2.line(frame, off_to_pixel((6.6,0)),  off_to_pixel((6.6,2.99)),  (0,255,0), 2)

prev = None
for deg in range(0, 181, 3):
    t = math.radians(deg)
    p = (6.75*math.cos(t), 1.575 + 6.75*math.sin(t))
    px = off_to_pixel(p)
    if prev:
        cv2.line(frame, prev, px, (0,255,0), 2)
    prev = px

# Hoop (red)
cv2.circle(frame, off_to_pixel((0, 1.575)), 6, (0,0,255), -1)

cv2.imwrite("media/checks/official_calibration_check.jpg", frame)
print("Saved media/checks/official_calibration_check.jpg")
print("CHECK: green arc on painted arc, cyan key on painted key, red dot on rim.")