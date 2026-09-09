"""Shared court projection helpers: feet anchors, homography, bounds."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from src.lens_undistort import load_intrinsics_from_calib, undistort_point

# FIBA 3x3 half-court (meters) — matches render_with_2dmap / calibration
COURT_X_MIN, COURT_X_MAX = -7.5, 7.5
COURT_Y_MIN, COURT_Y_MAX = 0.0, 11.0
# Soft margin for "valid for analytics" (bench / slight overshoot)
SOFT_X_PAD, SOFT_Y_PAD = 1.0, 1.0

# Feet: use lower band of bbox, not raw y2 (reduces shadow / paint bleed)
FEET_HEIGHT_FRAC = 0.96


def load_calibration(calibration_path: str | Path):
    """Return (H, K_or_None, dist_or_None, raw_dict)."""
    with open(calibration_path, "r") as f:
        data = json.load(f)
    H = np.array(data["homography_matrix"], dtype=np.float64)
    K, dist = load_intrinsics_from_calib(data)
    return H, K, dist, data


def load_homography(calibration_path: str | Path) -> np.ndarray:
    H, _, _, _ = load_calibration(calibration_path)
    return H


def px_to_meters(
    H: np.ndarray,
    pt: Sequence[float],
    K: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    ux, uy = undistort_point(pt, K, dist)
    v = np.array([ux, uy, 1.0], dtype=np.float64)
    t = H @ v
    w = t[2]
    if abs(w) < 1e-9:
        return float("nan"), float("nan")
    return float(t[0] / w), float(t[1] / w)


def meters_to_px(H_inv: np.ndarray, pt: Sequence[float]) -> Tuple[int, int]:
    v = np.array([float(pt[0]), float(pt[1]), 1.0], dtype=np.float64)
    t = H_inv @ v
    return int(round(t[0] / t[2])), int(round(t[1] / t[2]))


def feet_from_bbox(
    bbox: Sequence[float],
    frame_w: Optional[int] = None,
    frame_h: Optional[int] = None,
    height_frac: float = FEET_HEIGHT_FRAC,
) -> Tuple[Optional[list], bool]:
    """
    Ground contact proxy for a player bbox.
    Returns (feet_xy or None, is_usable).
    Rejects edge-clipped boxes where feet are not on the calibrated plane.
    """
    x1, y1, x2, y2 = map(float, bbox)
    if x2 <= x1 or y2 <= y1:
        return None, False

    # Truncated at image edge → feet not trustworthy
    if frame_h is not None and y2 >= frame_h - 2:
        return None, False
    if frame_w is not None and (x1 <= 1 or x2 >= frame_w - 2):
        return None, False

    fx = (x1 + x2) / 2.0
    fy = y1 + height_frac * (y2 - y1)
    return [int(round(fx)), int(round(fy))], True


def project_player_feet(
    H: np.ndarray,
    bbox: Sequence[float],
    frame_w: Optional[int] = None,
    frame_h: Optional[int] = None,
    K: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
) -> Tuple[Optional[list], Optional[list], bool]:
    """Returns (feet_px, court_meters, valid)."""
    feet, usable = feet_from_bbox(bbox, frame_w, frame_h)
    if not usable or feet is None:
        return None, None, False
    mx, my = px_to_meters(H, feet, K, dist)
    if not (math.isfinite(mx) and math.isfinite(my)):
        return feet, None, False
    inside = (
        COURT_X_MIN - SOFT_X_PAD <= mx <= COURT_X_MAX + SOFT_X_PAD
        and COURT_Y_MIN - SOFT_Y_PAD <= my <= COURT_Y_MAX + SOFT_Y_PAD
    )
    return feet, [round(mx, 3), round(my, 3)], inside


def project_ball_center(
    H: np.ndarray,
    center_px: Sequence[float],
    K: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
) -> Tuple[list, bool]:
    mx, my = px_to_meters(H, center_px, K, dist)
    if not (math.isfinite(mx) and math.isfinite(my)):
        return [0.0, 0.0], False
    return [round(mx, 3), round(my, 3)], True


def in_strict_court(meters: Sequence[float]) -> bool:
    x, y = meters
    return COURT_X_MIN <= x <= COURT_X_MAX and COURT_Y_MIN <= y <= COURT_Y_MAX
