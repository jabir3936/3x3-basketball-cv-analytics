import cv2, json, math
import numpy as np

cap = cv2.VideoCapture("media/footage/court_footage.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 99)
ret, frame = cap.read()
cap.release()

# Real on-image pixels only (1920x1080). Off-image clicks break H.
# Prefer: python scripts/02_calibration/refit_official_homography.py
# Or interactive: python scripts/02_calibration/calibrate_court_interactive.py
PIXEL_POINTS = [
    (350, 570),   # 1 key-left  @ baseline
    (857, 513),   # 2 key-right @ baseline
    (613, 784),   # 3 FT-left
    (1172, 685),  # 4 FT-right
    (1199, 472),  # 5 arc-right @ baseline (visible)
]

OFFICIAL_POINTS = [
    (-2.45, 0.0), (2.45, 0.0),
    (-2.45, 5.80), (2.45, 5.80),
    (6.60, 0.0),
]

H2, _ = cv2.findHomography(np.float32(PIXEL_POINTS), np.float32(OFFICIAL_POINTS))
H2_inv = np.linalg.inv(H2)

def off_to_pixel(p):
    v = np.array([p[0], p[1], 1.0], dtype=np.float32)
    t = H2_inv @ v
    return (int(round(t[0]/t[2])), int(round(t[1]/t[2])))

img = frame.copy()
for a, b in [((-7.5,0),(7.5,0)), ((7.5,0),(7.5,11)), ((7.5,11),(-7.5,11)), ((-7.5,11),(-7.5,0))]:
    cv2.line(img, off_to_pixel(a), off_to_pixel(b), (255,255,0), 2)
for a, b in [((-2.45,0),(2.45,0)), ((2.45,0),(2.45,5.8)), ((2.45,5.8),(-2.45,5.8)), ((-2.45,5.8),(-2.45,0))]:
    cv2.line(img, off_to_pixel(a), off_to_pixel(b), (0,255,255), 2)
cv2.line(img, off_to_pixel((-6.6,0)), off_to_pixel((-6.6,2.99)), (0,255,0), 2)
cv2.line(img, off_to_pixel((6.6,0)),  off_to_pixel((6.6,2.99)),  (0,255,0), 2)
prev = None
for deg in range(0, 181, 3):
    t = math.radians(deg)
    p = off_to_pixel((6.75*math.cos(t), 1.575 + 6.75*math.sin(t)))
    if prev: cv2.line(img, prev, p, (0,255,0), 2)
    prev = p
cv2.circle(img, off_to_pixel((0, 1.575)), 6, (0,0,255), -1)
cv2.imwrite("media/checks/official_calibration_checkv2.jpg", img)
print("Saved - check alignment")
print("Homography", H2.tolist())
with open("data/calibration/calibration_official_tuned.json", "w") as f:
    json.dump({"homography_matrix": H2.tolist()}, f, indent=2)
print("Saved data/calibration/calibration_official_tuned.json")