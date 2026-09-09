"""
Interactive court calibration: click landmarks, save H + overlay.

Usage (local machine with display):
  python scripts/02_calibration/calibrate_court_interactive.py

Keys:
  click = add point (order shown in window title)
  u     = undo last click
  s     = fit + save when enough points
  q     = quit without saving
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "media/footage/court_footage.mp4"
OUT_CALIB = ROOT / "data/calibration/calibration_official_tuned.json"
OUT_IMG = ROOT / "media/checks/official_calibration_check_interactive.jpg"
FRAME_INDEX = 99

# Click in this order (or subset of first N>=4)
LANDMARKS = [
    ("key-left baseline", (-2.45, 0.0)),
    ("key-right baseline", (2.45, 0.0)),
    ("FT-left", (-2.45, 5.80)),
    ("FT-right", (2.45, 5.80)),
    ("arc-left baseline", (-6.60, 0.0)),
    ("arc-right baseline", (6.60, 0.0)),
    ("sideline-left near FT-y", (-7.50, 5.80)),
    ("sideline-right near FT-y", (7.50, 5.80)),
]

clicks = []


def redraw(base):
    img = base.copy()
    for i, (x, y) in enumerate(clicks):
        cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
        name = LANDMARKS[i][0] if i < len(LANDMARKS) else str(i)
        cv2.putText(img, f"{i+1}:{name}", (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    nxt = LANDMARKS[len(clicks)][0] if len(clicks) < len(LANDMARKS) else "DONE (press s)"
    cv2.putText(img, f"Next: {nxt} | u=undo s=save q=quit", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(LANDMARKS):
        clicks.append((x, y))
        print(f"  click {len(clicks)}: ({x},{y}) -> {LANDMARKS[len(clicks)-1]}")


def off_to_pixel(H_inv, p):
    v = np.array([p[0], p[1], 1.0], dtype=np.float64)
    t = H_inv @ v
    return (int(round(t[0] / t[2])), int(round(t[1] / t[2])))


def draw_overlay(frame, H):
    H_inv = np.linalg.inv(H)
    img = frame.copy()
    for a, b in [((-7.5, 0), (7.5, 0)), ((7.5, 0), (7.5, 11)),
                 ((7.5, 11), (-7.5, 11)), ((-7.5, 11), (-7.5, 0))]:
        cv2.line(img, off_to_pixel(H_inv, a), off_to_pixel(H_inv, b), (255, 255, 0), 2)
    for a, b in [((-2.45, 0), (2.45, 0)), ((2.45, 0), (2.45, 5.8)),
                 ((2.45, 5.8), (-2.45, 5.8)), ((-2.45, 5.8), (-2.45, 0))]:
        cv2.line(img, off_to_pixel(H_inv, a), off_to_pixel(H_inv, b), (0, 255, 255), 2)
    prev = None
    for deg in range(0, 181, 3):
        t = math.radians(deg)
        p = off_to_pixel(H_inv, (6.75 * math.cos(t), 1.575 + 6.75 * math.sin(t)))
        if prev:
            cv2.line(img, prev, p, (0, 255, 0), 2)
        prev = p
    cv2.circle(img, off_to_pixel(H_inv, (0, 1.575)), 6, (0, 0, 255), -1)
    return img


def main():
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise SystemExit("Could not read frame")

    win = "calibrate_court"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        img = redraw(frame)
        cv2.imshow(win, img)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("u") and clicks:
            clicks.pop()
        if key == ord("s"):
            if len(clicks) < 4:
                print("Need at least 4 clicks")
                continue
            src = np.float32(clicks)
            dst = np.float32([LANDMARKS[i][1] for i in range(len(clicks))])
            H, _ = cv2.findHomography(src, dst, method=0)
            overlay = draw_overlay(frame, H)
            for i, (x, y) in enumerate(clicks):
                cv2.circle(overlay, (x, y), 6, (0, 0, 255), -1)
            cv2.imwrite(str(OUT_IMG), overlay)
            payload = {
                "homography_matrix": H.tolist(),
                "pixel_points": clicks,
                "official_points_m": [LANDMARKS[i][1] for i in range(len(clicks))],
                "calibration_frame_index": FRAME_INDEX,
            }
            with open(OUT_CALIB, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"Saved {OUT_CALIB} and {OUT_IMG}")
            cv2.imshow("overlay preview", overlay)
            cv2.waitKey(0)
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
