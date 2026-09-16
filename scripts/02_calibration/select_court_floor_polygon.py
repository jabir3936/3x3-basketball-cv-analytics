"""Click the playable-court boundary used to reject off-court player detections.

Run after the interactive calibration, from a local VS Code/Cursor terminal:
    python scripts/02_calibration/select_court_floor_polygon.py

Click 4–12 points just INSIDE the playable court boundary, in order around the
floor.  Do not include LED boards, photographers, referee areas, or the bench.
Keys: u=undo, s=save (needs at least four points), q=quit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.court_projection import load_calibration  # noqa: E402

VIDEO = ROOT / "media/footage/court_footage.mp4"
CALIBRATION = ROOT / "data/calibration/calibration_official_tuned.json"
OUTPUT = ROOT / "data/calibration/court_floor_polygon.json"
OUT_IMAGE = ROOT / "media/checks/court_floor_polygon_check.jpg"
points: list[tuple[int, int]] = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"point {len(points)}: ({x}, {y})")


def draw(frame):
    img = frame.copy()
    if points:
        poly = np.array(points, dtype=np.int32)
        cv2.polylines(img, [poly], len(points) >= 3, (0, 255, 255), 2)
        for i, p in enumerate(points, start=1):
            cv2.circle(img, p, 5, (0, 0, 255), -1)
            cv2.putText(img, str(i), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, "Click playable floor boundary | u=undo s=save q=quit", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return img


def main():
    _, K, dist, calib = load_calibration(CALIBRATION)
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(calib.get("calibration_frame_index", 99)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not read calibration frame")
    if K is not None and dist is not None:
        frame = cv2.undistort(frame, K, dist)

    window = "select_court_floor_polygon"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        cv2.imshow(window, draw(frame))
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("u") and points:
            points.pop()
        if key == ord("s"):
            if len(points) < 4:
                print("Need at least four points")
                continue
            OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            with OUTPUT.open("w") as f:
                json.dump({
                    "calibration_frame_id": int(calib.get("calibration_frame_index", 99)) + 1,
                    "coordinate_space": "undistorted_calibration_frame_px",
                    "polygon_px": points,
                }, f, indent=2)
            cv2.imwrite(str(OUT_IMAGE), draw(frame))
            print(f"Saved {OUTPUT}\nSaved {OUT_IMAGE}")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
