import cv2
import json
import math
import numpy as np

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
VIDEO_PATH = "media/footage/court_footage.mp4"
FRAME_ID = 100

# Click in THIS exact order
LABELS = [
    "1 key-left @ baseline",
    "2 key-right @ baseline",
    "3 free-throw-left",
    "4 free-throw-right",
    "5 arc-left @ baseline",
    "6 arc-right @ baseline",
]

OFFICIAL_POINTS = [
    (-2.45, 0.0),
    ( 2.45, 0.0),
    (-2.45, 5.80),
    ( 2.45, 5.80),
    (-6.60, 0.0),
    ( 6.60, 0.0),
]

# ---------------------------------------------------------
# LOAD FRAME
# ---------------------------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_ID - 1)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError(f"Could not read frame {FRAME_ID} from {VIDEO_PATH}")

display = frame.copy()
clicked = []
WIN = "Click 6 landmarks in order (u = undo, r = reset, q = done)"

def redraw():
    global display
    display = frame.copy()
    for i, (x, y) in enumerate(clicked, 1):
        cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(display, str(i), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow(WIN, display)

def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 6:
        clicked.append((x, y))
        print(f"Point {len(clicked)} [{LABELS[len(clicked) - 1]}]: ({x}, {y})")
        redraw()

cv2.namedWindow(WIN)
cv2.setMouseCallback(WIN, mouse_cb)
redraw()

while True:
    k = cv2.waitKey(20) & 0xFF
    if k == ord('u') and clicked:
        clicked.pop()
        print(f"Undone. {len(clicked)} points kept.")
        redraw()
    elif k == ord('r'):
        clicked = []
        print("Reset.")
        redraw()
    elif k == ord('q'):
        break

cv2.destroyAllWindows()

if len(clicked) != 6:
    print(f"Need 6 points, got {len(clicked)}. Aborting.")
else:
    print("\nPIXEL_POINTS =", clicked)

    # -----------------------------------------------------
    # COMPUTE NEW HOMOGRAPHY (pixel -> official meters)
    # -----------------------------------------------------
    H2, _ = cv2.findHomography(
        np.float32(clicked),
        np.float32(OFFICIAL_POINTS)
    )

    with open("data/calibration/calibration_official.json", "w") as f:
        json.dump({"homography_matrix": H2.tolist()}, f, indent=2)
    print("Saved data/calibration/calibration_official.json")

    # -----------------------------------------------------
    # VALIDATION OVERLAY
    # -----------------------------------------------------
    H2_inv = np.linalg.inv(H2)

    def off_to_pixel(p):
        v = np.array([p[0], p[1], 1.0], dtype=np.float32)
        t = H2_inv @ v
        return (int(round(t[0] / t[2])), int(round(t[1] / t[2])))

    img = frame.copy()

    # Boundary (yellow)
    for a, b in [((-7.5, 0), (7.5, 0)), ((7.5, 0), (7.5, 11)),
                 ((7.5, 11), (-7.5, 11)), ((-7.5, 11), (-7.5, 0))]:
        cv2.line(img, off_to_pixel(a), off_to_pixel(b), (255, 255, 0), 2)

    # Key (cyan)
    for a, b in [((-2.45, 0), (2.45, 0)), ((2.45, 0), (2.45, 5.8)),
                 ((2.45, 5.8), (-2.45, 5.8)), ((-2.45, 5.8), (-2.45, 0))]:
        cv2.line(img, off_to_pixel(a), off_to_pixel(b), (0, 255, 255), 2)

    # Arc straights (green)
    cv2.line(img, off_to_pixel((-6.6, 0)), off_to_pixel((-6.6, 2.99)), (0, 255, 0), 2)
    cv2.line(img, off_to_pixel((6.6, 0)),  off_to_pixel((6.6, 2.99)),  (0, 255, 0), 2)

    # Arc (green)
    prev = None
    for deg in range(0, 181, 3):
        t = math.radians(deg)
        p = off_to_pixel((6.75 * math.cos(t), 1.575 + 6.75 * math.sin(t)))
        if prev:
            cv2.line(img, prev, p, (0, 255, 0), 2)
        prev = p

    # Hoop (red)
    cv2.circle(img, off_to_pixel((0, 1.575)), 6, (0, 0, 255), -1)

    cv2.imwrite("media/checks/official_calibration_check.jpg", img)
    print("Saved media/checks/official_calibration_check.jpg")
    print("CHECK: green arc on painted arc, cyan key on painted key, red dot on rim.")