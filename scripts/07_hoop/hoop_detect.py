import cv2, json, math
import numpy as np

with open("data/calibration/calibration_new.json") as f:
    H = np.array(json.load(f)["homography_matrix"], dtype=np.float32)
H_inv = np.linalg.inv(H)

def pixel_to_court(px):
    v = np.array([px[0], px[1], 1.0], dtype=np.float32)
    t = H @ v
    return [float(t[0]/t[2]), float(t[1]/t[2])]

def court_to_pixel(c):
    v = np.array([c[0], c[1], 1.0], dtype=np.float32)
    t = H_inv @ v
    return (int(t[0]/t[2]), int(t[1]/t[2]))

# Pixels traced along the PAINTED WHITE 3-point arc (adjust if needed)
ARC_PIXELS = [
    (375, 570), (205, 627), (300, 700), (475, 762), (700, 790),
    (900, 788), (1100, 762), (1300, 700), (1408, 610), (1440, 540)
]

pts = np.array([pixel_to_court(p) for p in ARC_PIXELS])
x, y = pts[:, 0], pts[:, 1]

# Least-squares circle fit (Kasa method)
A = np.c_[2*x, 2*y, np.ones(len(x))]
b = x**2 + y**2
ux, uy, c = np.linalg.lstsq(A, b, rcond=None)[0]
radius = math.sqrt(c + ux**2 + uy**2)

hoop = [ux, uy]
print("Fitted HOOP court coord :", [round(ux,3), round(uy,3)])
print("Fitted radius (want ~6.75):", round(radius,3))

# Overlay check
cap = cv2.VideoCapture("media/footage/court_footage.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 99)
ret, frame = cap.read()
cap.release()

hp = court_to_pixel(hoop)
cv2.circle(frame, hp, 8, (0,0,255), -1)
cv2.putText(frame, "HOOP", (hp[0]+10, hp[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
for deg in range(0, 360, 3):
    a = math.radians(deg)
    pt = court_to_pixel([hoop[0]+radius*math.cos(a), hoop[1]+radius*math.sin(a)])
    if 0 <= pt[0] < frame.shape[1] and 0 <= pt[1] < frame.shape[0]:
        cv2.circle(frame, pt, 3, (0,255,0), -1)
cv2.imwrite("media/checks/hoop_verification2.jpg", frame)
print("Saved media/checks/hoop_verification2.jpg")