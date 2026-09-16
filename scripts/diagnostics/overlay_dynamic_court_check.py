"""Draw the dynamic per-frame court projection for visual calibration QA."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.court_projection import load_calibration  # noqa: E402

VIDEO = ROOT / "media/footage/court_footage.mp4"
CALIB = ROOT / "data/calibration/calibration_official_tuned.json"
MOTION = ROOT / "data/calibration/camera_motion_to_calibration.json"
OUT = ROOT / "media/checks"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="400,800,1600")
    args = ap.parse_args()
    frame_ids = [int(v) for v in args.frames.split(",")]
    H, K, dist, _ = load_calibration(CALIB)
    with open(MOTION) as f:
        motion = json.load(f)["frames"]
    cap = cv2.VideoCapture(str(VIDEO))

    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid - 1)
        ok, frame = cap.read()
        if not ok:
            continue
        if K is not None and dist is not None:
            frame = cv2.undistort(frame, K, dist)
        T = np.array(motion[str(fid)]["transform_to_calibration"], dtype=float)
        H_inv = np.linalg.inv(H @ T)  # court -> current undistorted frame

        def p(x, y):
            q = H_inv @ np.array([x, y, 1.0])
            return int(round(q[0] / q[2])), int(round(q[1] / q[2]))

        for a, b in [((-7.5, 0), (7.5, 0)), ((7.5, 0), (7.5, 11)),
                     ((7.5, 11), (-7.5, 11)), ((-7.5, 11), (-7.5, 0))]:
            cv2.line(frame, p(*a), p(*b), (255, 255, 0), 2)
        for a, b in [((-2.45, 0), (2.45, 0)), ((2.45, 0), (2.45, 5.8)),
                     ((2.45, 5.8), (-2.45, 5.8)), ((-2.45, 5.8), (-2.45, 0))]:
            cv2.line(frame, p(*a), p(*b), (0, 255, 255), 2)
        prev = None
        for deg in range(0, 181, 3):
            t = math.radians(deg)
            cur = p(6.75 * math.cos(t), 1.575 + 6.75 * math.sin(t))
            if prev:
                cv2.line(frame, prev, cur, (0, 255, 0), 2)
            prev = cur
        out = OUT / f"dynamic_court_check_frame_{fid:04d}.jpg"
        cv2.imwrite(str(out), frame)
        print(f"Saved {out}")
    cap.release()


if __name__ == "__main__":
    main()
