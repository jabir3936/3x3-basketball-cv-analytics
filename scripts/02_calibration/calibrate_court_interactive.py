"""Create and validate a court homography from manually clicked landmarks.

Run locally (requires a display):
    python scripts/02_calibration/calibrate_court_interactive.py

Click the named landmarks in order.  Use ``n`` to skip a landmark that is not
visible; use ``u`` to undo.  The first six are fit points, while the final four
are held out and used only to validate the result.  A failed validation is not
saved unless you revisit the fit landmarks.

The default is no lens correction.  Do not set --k1 until it has been measured
from this camera; a guessed distortion coefficient can worsen the map.
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
from src.lens_undistort import default_camera_matrix, dist_from_k1  # noqa: E402

VIDEO = ROOT / "media/footage/court_footage.mp4"
OUT_CALIB = ROOT / "data/calibration/calibration_official_tuned.json"
OUT_IMG = ROOT / "media/checks/official_calibration_check_interactive.jpg"
OUT_REPORT = ROOT / "data/calibration/calibration_validation_report.json"
FRAME_INDEX = 900

# Fit points should span the visible play area. Hold-out points must NOT be
# used to fit H; they are the only meaningful measure of calibration accuracy.
LANDMARKS = [
    ("FIT: key-left baseline", (-2.45, 0.00), "fit"),
    ("FIT: key-right baseline", (2.45, 0.00), "fit"),
    ("FIT: key-left free-throw line", (-2.45, 5.80), "fit"),
    ("FIT: key-right free-throw line", (2.45, 5.80), "fit"),
    ("FIT: right 3pt baseline", (6.60, 0.00), "fit"),
    ("FIT: right 3pt straight/arc join", (6.60, 2.99), "fit"),
    ("CHECK: left 3pt baseline", (-6.60, 0.00), "check"),
    ("CHECK: left 3pt straight/arc join", (-6.60, 2.99), "check"),
    ("CHECK: left sideline/baseline corner", (-7.50, 0.00), "check"),
    ("CHECK: right sideline/baseline corner", (7.50, 0.00), "check"),
]
MAX_MEAN_ERROR_M = 0.50
MAX_POINT_ERROR_M = 0.75
clicks: list[tuple[int, int] | None] = []


def to_court(H, pixel):
    p = np.array([pixel[0], pixel[1], 1.0], dtype=np.float64)
    q = H @ p
    return float(q[0] / q[2]), float(q[1] / q[2])


def court_to_pixel(H_inv, point):
    p = np.array([point[0], point[1], 1.0], dtype=np.float64)
    q = H_inv @ p
    return int(round(q[0] / q[2])), int(round(q[1] / q[2]))


def draw_overlay(frame, H):
    H_inv = np.linalg.inv(H)
    image = frame.copy()
    for a, b in [((-7.5, 0), (7.5, 0)), ((7.5, 0), (7.5, 11)),
                 ((7.5, 11), (-7.5, 11)), ((-7.5, 11), (-7.5, 0))]:
        cv2.line(image, court_to_pixel(H_inv, a), court_to_pixel(H_inv, b), (255, 255, 0), 2)
    for a, b in [((-2.45, 0), (2.45, 0)), ((2.45, 0), (2.45, 5.8)),
                 ((2.45, 5.8), (-2.45, 5.8)), ((-2.45, 5.8), (-2.45, 0))]:
        cv2.line(image, court_to_pixel(H_inv, a), court_to_pixel(H_inv, b), (0, 255, 255), 2)
    previous = None
    for degree in range(0, 181, 3):
        angle = math.radians(degree)
        current = court_to_pixel(H_inv, (6.75 * math.cos(angle), 1.575 + 6.75 * math.sin(angle)))
        if previous:
            cv2.line(image, previous, current, (0, 255, 0), 2)
        previous = current
    return image


def redraw(frame):
    image = frame.copy()
    for i, point in enumerate(clicks):
        name, _, purpose = LANDMARKS[i]
        color = (0, 255, 0) if purpose == "fit" else (0, 165, 255)
        if point is None:
            cv2.putText(image, f"{i + 1}: skipped", (20, 80 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        else:
            cv2.circle(image, point, 6, color, -1)
            cv2.putText(image, f"{i + 1}: {name}", (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    next_name = LANDMARKS[len(clicks)][0] if len(clicks) < len(LANDMARKS) else "all landmarks entered: press s to validate"
    cv2.putText(image, f"Next: {next_name}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
    cv2.putText(image, "click=record | n=skip | u=undo | s=validate/save | q=quit", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k1", type=float, default=0.0, help="Measured radial lens coefficient; default 0 disables correction.")
    parser.add_argument("--fov", type=float, default=85.0, help="Used only with a measured --k1.")
    args = parser.parse_args()
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not read calibration frame")

    K = dist = None
    if abs(args.k1) > 1e-9:
        K = default_camera_matrix(frame.shape[1], frame.shape[0], args.fov)
        dist = dist_from_k1(args.k1)
        frame = cv2.undistort(frame, K, dist)
        print("Using supplied lens correction; make sure k1 was measured, not guessed.")

    window = "calibrate_court_interactive"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(LANDMARKS):
            clicks.append((x, y))
            print(f"{len(clicks)}. {LANDMARKS[len(clicks) - 1][0]} = ({x}, {y})")

    cv2.setMouseCallback(window, on_mouse)
    while True:
        cv2.imshow(window, redraw(frame))
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("u") and clicks:
            clicks.pop()
        if key == ord("n") and len(clicks) < len(LANDMARKS):
            print(f"Skipped: {LANDMARKS[len(clicks)][0]}")
            clicks.append(None)
        if key != ord("s"):
            continue

        entered = list(enumerate(clicks))
        fit = [(point, LANDMARKS[i]) for i, point in entered if point is not None and LANDMARKS[i][2] == "fit"]
        checks = [(point, LANDMARKS[i]) for i, point in entered if point is not None and LANDMARKS[i][2] == "check"]
        if len(fit) < 4:
            print("Need at least four visible FIT landmarks")
            continue
        if len(checks) < 2:
            print("Need at least two visible CHECK landmarks for validation")
            continue

        src = np.float32([pixel for pixel, _ in fit])
        dst = np.float32([landmark[1] for _, landmark in fit])
        method = cv2.RANSAC if len(fit) >= 5 else 0
        H, _ = cv2.findHomography(src, dst, method, 2.0)
        if H is None:
            print("Homography fit failed; re-check landmark clicks")
            continue
        errors = []
        for pixel, landmark in checks:
            projected = to_court(H, pixel)
            target = landmark[1]
            error = math.dist(projected, target)
            errors.append({"name": landmark[0], "pixel": pixel, "expected_m": target, "projected_m": projected, "error_m": error})
            print(f"{landmark[0]}: error={error:.3f}m")
        mean_error = float(np.mean([entry["error_m"] for entry in errors]))
        worst_error = float(max(entry["error_m"] for entry in errors))
        passed = mean_error <= MAX_MEAN_ERROR_M and worst_error <= MAX_POINT_ERROR_M
        report = {"passed": passed, "mean_error_m": mean_error, "worst_error_m": worst_error, "thresholds_m": {"mean": MAX_MEAN_ERROR_M, "worst": MAX_POINT_ERROR_M}, "checks": errors}
        OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
        with OUT_REPORT.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"Validation mean={mean_error:.3f}m, worst={worst_error:.3f}m => {'PASS' if passed else 'FAIL'}")
        if not passed:
            print(f"Not saved. Inspect {OUT_REPORT}, undo/re-click FIT points, and try again.")
            continue

        overlay = draw_overlay(frame, H)
        for point, _ in checks:
            cv2.circle(overlay, point, 7, (0, 165, 255), -1)
        cv2.imwrite(str(OUT_IMG), overlay)
        payload = {
            "homography_matrix": H.tolist(),
            "fit_pixel_points": [pixel for pixel, _ in fit],
            "fit_official_points_m": [landmark[1] for _, landmark in fit],
            "validation": report,
            "calibration_frame_index": FRAME_INDEX,
            "notes": "Validated with held-out landmarks; points are undistorted if k1 is supplied.",
        }
        if K is not None:
            payload.update({"fov_deg": args.fov, "k1": args.k1, "camera_matrix": K.tolist(), "dist_coeffs": dist.tolist()})
        with OUT_CALIB.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"PASS: saved {OUT_CALIB}, {OUT_REPORT}, and {OUT_IMG}")
        break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
