import cv2
import math
import numpy as np
import json

# Load homography
with open("data/calibration/calibration_new.json", "r") as f:
    H = np.array(json.load(f)["homography_matrix"], dtype=np.float32)
H_inv = np.linalg.inv(H)

def court_to_pixel(x, y):
    vec = np.array([x, y, 1.0], dtype=np.float32)
    res = np.dot(H_inv, vec)
    return (int(res[0]/res[2]), int(res[1]/res[2]))

# Read a frame where the court is visible
cap = cv2.VideoCapture("media/footage/court_footage.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 100) 
ret, frame = cap.read()
cap.release()

# 1. Draw Hoop at [0, 0]
hoop_px = court_to_pixel(200, 800)
cv2.circle(frame, hoop_px, 8, (0, 0, 255), -1) # Red dot
cv2.putText(frame, "HOOP [0,0]", (hoop_px[0]+10, hoop_px[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# 2. Draw 6.75m 3-Point Arc
for deg in range(0, 181, 3): # 0 to 180 degrees covers the court side of the arc
    a = math.radians(deg)
    x = 6.75 * math.cos(a)
    y = 6.75 * math.sin(a)
    
    px = court_to_pixel(x, y)
    if 0 <= px[0] < frame.shape[1] and 0 <= px[1] < frame.shape[0]:
        cv2.circle(frame, px, 3, (0, 255, 0), -1) # Green dots

cv2.imwrite("media/checks/hoop_verification.jpg", frame)
print("Saved media/checks/hoop_verification.jpg - Open it to see the red dot on the rim and green arc!")