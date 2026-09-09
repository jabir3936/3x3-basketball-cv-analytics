"""
Refit official homography from verified on-court pixel landmarks.
Optionally fit in undistorted space (set k1) to reduce wide-angle warp.

Usage:
  python scripts/02_calibration/refit_official_homography.py
  python scripts/02_calibration/refit_official_homography.py --k1 -0.18
"""
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

from src.lens_undistort import default_camera_matrix, dist_from_k1, undistort_points  # noqa: E402

VIDEO = ROOT / "media/footage/court_footage.mp4"
OUT_CALIB = ROOT / "data/calibration/calibration_official_tuned.json"
OUT_IMG = ROOT / "media/checks/official_calibration_checkv2.jpg"
FRAME_INDEX = 99

PIXEL_POINTS = [
    (350, 570),
    (857, 513),
    (613, 784),
    (1172, 685),
    (1199, 472),
]

OFFICIAL_POINTS = [
    (-2.45, 0.0),
    (2.45, 0.0),
    (-2.45, 5.80),
    (2.45, 5.80),
    (6.60, 0.0),
]


def off_to_pixel(H_inv, p, K=None, dist=None):
    """Meters -> distorted pixel (for drawing on original frame)."""
    v = np.array([p[0], p[1], 1.0], dtype=np.float64)
    t = H_inv @ v
    ux, uy = t[0] / t[2], t[1] / t[2]
    if K is None or dist is None:
        return (int(round(ux)), int(round(uy)))
    # undistorted -> distorted via distortPoints isn't in older cv2; approximate with
    # iterative undistort inversion is heavy — draw in undistorted space on undistorted img instead.
    pts = np.array([[[ux, uy]]], dtype=np.float64)
    # Project as if ux,uy are normalized... use cv2.projectPoints from unit plane:
    # Simpler path: caller draws on undistorted image when K is set.
    return (int(round(ux)), int(round(uy)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k1", type=float, default=-0.16,
                    help="radial distortion coeff (0=off). Negative = barrel (wide-angle).")
    ap.add_argument("--fov", type=float, default=85.0)
    args = ap.parse_args()

    assert len(PIXEL_POINTS) == len(OFFICIAL_POINTS) and len(PIXEL_POINTS) >= 4
    for i, (x, y) in enumerate(PIXEL_POINTS):
        if not (0 <= x < 1920 and 0 <= y < 1080):
            raise SystemExit(f"PIXEL_POINTS[{i}]={(x,y)} off-image")

    use_undistort = abs(args.k1) > 1e-6
    K = dist = None
    src_pts = np.float32(PIXEL_POINTS)
    if use_undistort:
        K = default_camera_matrix(1920, 1080, args.fov)
        dist = dist_from_k1(args.k1)
        src_pts = undistort_points(PIXEL_POINTS, K, dist).astype(np.float32)

    dst = np.float32(OFFICIAL_POINTS)
    H, _ = cv2.findHomography(src_pts, dst, method=0)
    if H is None:
        raise SystemExit("findHomography failed")
    H_inv = np.linalg.inv(H)

    errs = []
    for px, off in zip(PIXEL_POINTS, OFFICIAL_POINTS):
        if use_undistort:
            u = undistort_points([px], K, dist)[0]
            v = np.array([u[0], u[1], 1.0], dtype=np.float64)
        else:
            v = np.array([px[0], px[1], 1.0], dtype=np.float64)
        t = H @ v
        mx, my = t[0] / t[2], t[1] / t[2]
        e = math.hypot(mx - off[0], my - off[1])
        errs.append(e)
        print(f"  pixel {px} -> ({mx:.3f},{my:.3f}) expect {off} err={e:.4f}m")
    print(f"Mean reprojection error: {sum(errs)/len(errs):.4f} m  (k1={args.k1})")

    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise SystemExit("Could not read calibration frame")

    # Draw overlay on undistorted frame when using k1 (fair visual check)
    if use_undistort:
        img = cv2.undistort(frame, K, dist)
        def to_px(p):
            v = np.array([p[0], p[1], 1.0], dtype=np.float64)
            t = H_inv @ v
            return (int(round(t[0] / t[2])), int(round(t[1] / t[2])))
    else:
        img = frame.copy()
        def to_px(p):
            return off_to_pixel(H_inv, p)

    for a, b in [((-7.5, 0), (7.5, 0)), ((7.5, 0), (7.5, 11)),
                 ((7.5, 11), (-7.5, 11)), ((-7.5, 11), (-7.5, 0))]:
        cv2.line(img, to_px(a), to_px(b), (255, 255, 0), 2)
    for a, b in [((-2.45, 0), (2.45, 0)), ((2.45, 0), (2.45, 5.8)),
                 ((2.45, 5.8), (-2.45, 5.8)), ((-2.45, 5.8), (-2.45, 0))]:
        cv2.line(img, to_px(a), to_px(b), (0, 255, 255), 2)
    cv2.line(img, to_px((-6.6, 0)), to_px((-6.6, 2.99)), (0, 255, 0), 2)
    cv2.line(img, to_px((6.6, 0)), to_px((6.6, 2.99)), (0, 255, 0), 2)
    prev = None
    for deg in range(0, 181, 3):
        t = math.radians(deg)
        p = to_px((6.75 * math.cos(t), 1.575 + 6.75 * math.sin(t)))
        if prev:
            cv2.line(img, prev, p, (0, 255, 0), 2)
        prev = p
    cv2.circle(img, to_px((0, 1.575)), 6, (0, 0, 255), -1)

    for px, off in zip(PIXEL_POINTS, OFFICIAL_POINTS):
        if use_undistort:
            u = undistort_points([px], K, dist)[0]
            draw_pt = (int(round(u[0])), int(round(u[1])))
        else:
            draw_pt = (int(px[0]), int(px[1]))
        cv2.circle(img, draw_pt, 7, (0, 0, 255), -1)
        cv2.putText(img, f"{off}", (draw_pt[0] + 8, draw_pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    OUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_IMG), img)

    payload = {
        "homography_matrix": H.tolist(),
        "pixel_points": PIXEL_POINTS,
        "official_points_m": OFFICIAL_POINTS,
        "calibration_frame_index": FRAME_INDEX,
        "mean_reprojection_error_m": sum(errs) / len(errs),
        "fov_deg": args.fov,
        "k1": args.k1,
        "notes": "H is fit in undistorted space when k1!=0. Projection undistorts pixels first.",
    }
    if use_undistort:
        payload["camera_matrix"] = K.tolist()
        payload["dist_coeffs"] = dist.tolist()

    with open(OUT_CALIB, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {OUT_CALIB}")
    print(f"Saved {OUT_IMG}")


if __name__ == "__main__":
    main()
