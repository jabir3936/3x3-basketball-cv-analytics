"""Estimate a per-frame image transform back to the calibration frame.

The broadcast camera pans and zooms.  A static court homography is only valid
for the frame it was clicked on, so this writes transforms that map each
undistorted video frame into the calibration frame's undistorted coordinates.

Run from the project root:
    python scripts/03_processing/estimate_camera_motion.py
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
OUTPUT = ROOT / "data/calibration/camera_motion_to_calibration.json"
COURT_FLOOR = ROOT / "data/calibration/court_floor_polygon.json"
SAMPLE_EVERY = 15  # half-second at 30 fps; interpolated for intervening frames


def _frame(cap: cv2.VideoCapture, index0: int, K, dist):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index0)
    ok, img = cap.read()
    if not ok:
        return None
    if K is not None and dist is not None:
        img = cv2.undistort(img, K, dist)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def main():
    H, K, dist, calibration = load_calibration(CALIBRATION)
    calibration_index = int(calibration.get("calibration_frame_index", 99))
    cap = cv2.VideoCapture(str(VIDEO))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    base = _frame(cap, calibration_index, K, dist)
    if base is None:
        raise SystemExit("Could not read calibration frame")
    if not COURT_FLOOR.exists():
        raise SystemExit(
            f"Missing {COURT_FLOOR}. Run select_court_floor_polygon.py before this step."
        )
    with open(COURT_FLOOR) as f:
        floor_polygon = json.load(f).get("polygon_px", [])
    if len(floor_polygon) < 3:
        raise SystemExit(f"Invalid court-floor polygon: {COURT_FLOOR}")

    # Only court-plane features are valid for a court-plane homography.  Exclude
    # advertising boards, crowd, referees, and bench areas even if they offer
    # more visually distinctive ORB keypoints.
    mask = np.zeros(base.shape, np.uint8)
    cv2.fillPoly(mask, [np.array(floor_polygon, dtype=np.int32)], 255)

    orb = cv2.ORB_create(nfeatures=3000, fastThreshold=12)
    base_kp, base_desc = orb.detectAndCompute(base, mask)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    keyframes = {}
    last_good = np.eye(3, dtype=np.float64)

    for index0 in range(0, frame_count, SAMPLE_EVERY):
        current = _frame(cap, index0, K, dist)
        if current is None:
            continue
        # The reference descriptors are court-only.  The current frame cannot
        # reuse the reference mask because the camera has moved.
        kp, desc = orb.detectAndCompute(current, None)
        good = []
        if desc is not None and base_desc is not None:
            pairs = matcher.knnMatch(desc, base_desc, k=2)
            good = [a for a, b in pairs if a.distance < 0.70 * b.distance]

        # current -> calibration frame, so H_current = H_calibration @ T
        T = None
        inliers = 0
        if len(good) >= 12:
            src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([base_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            T, mask_in = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
            inliers = int(mask_in.sum()) if mask_in is not None else 0
            if T is not None and inliers >= 12:
                last_good = T
            else:
                T = None
        if T is None:
            T = last_good

        keyframes[index0 + 1] = {
            "transform_to_calibration": T.tolist(),
            "feature_matches": len(good),
            "inliers": inliers,
            "fallback": inliers < 12,
        }
        if (index0 + 1) % 300 == 0:
            print(f"{index0 + 1}/{frame_count}")

    cap.release()
    # Interpolate the projective matrices between sampled keyframes.  This is
    # appropriate for the smooth pan/zoom motion in this broadcast; each
    # transform still comes from a robust feature match, rather than assuming
    # that the camera never moves.
    ids = sorted(keyframes)
    results = {}
    for frame_id in range(1, frame_count + 1):
        left = max((i for i in ids if i <= frame_id), default=ids[0])
        right = min((i for i in ids if i >= frame_id), default=ids[-1])
        a = np.array(keyframes[left]["transform_to_calibration"], dtype=float)
        b = np.array(keyframes[right]["transform_to_calibration"], dtype=float)
        t = 0.0 if left == right else (frame_id - left) / (right - left)
        T = (1.0 - t) * a + t * b
        T /= T[2, 2]
        nearest = left if abs(frame_id - left) <= abs(right - frame_id) else right
        results[str(frame_id)] = {
            "transform_to_calibration": T.tolist(),
            "source_frame": nearest,
            "feature_matches": keyframes[nearest]["feature_matches"],
            "inliers": keyframes[nearest]["inliers"],
            "fallback": keyframes[nearest]["fallback"],
        }

    payload = {
        "calibration_frame_id": calibration_index + 1,
        "frame_count": frame_count,
        "description": "Maps undistorted current-frame pixels to undistorted calibration-frame pixels.",
        "sample_every_frames": SAMPLE_EVERY,
        "frames": results,
    }
    with open(OUTPUT, "w") as f:
        json.dump(payload, f)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
