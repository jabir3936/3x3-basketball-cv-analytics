import json
import numpy as np

with open("data/calibration/calibration_new.json", "r") as f:
    H = np.array(json.load(f)["homography_matrix"], dtype=np.float32)

def pixel_to_court(px):
    x, y = px
    v = np.array([x, y, 1.0], dtype=np.float32)
    t = np.dot(H, v)
    return [round(float(t[0] / t[2]), 3), round(float(t[1] / t[2]), 3)]

HOOP_PIXEL = (200, 800)   # <-- replace with the pixel you read
print("Hoop court coordinate:", pixel_to_court(HOOP_PIXEL))