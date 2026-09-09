"""Approximate lens undistortion for wide-angle court footage."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


def default_camera_matrix(frame_w: int = 1920, frame_h: int = 1080, fov_deg: float = 85.0) -> np.ndarray:
    """Pinhole K from horizontal FOV guess (wide broadcast-ish)."""
    fx = (frame_w / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    fy = fx
    cx, cy = frame_w / 2.0, frame_h / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def dist_from_k1(k1: float) -> np.ndarray:
    # [k1, k2, p1, p2, k3]
    return np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def undistort_points(
    pts: Sequence[Sequence[float]],
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    arr = np.array(pts, dtype=np.float64).reshape(-1, 1, 2)
    und = cv2.undistortPoints(arr, K, dist, P=K)
    return und.reshape(-1, 2)


def undistort_point(
    pt: Sequence[float],
    K: Optional[np.ndarray],
    dist: Optional[np.ndarray],
) -> Tuple[float, float]:
    if K is None or dist is None:
        return float(pt[0]), float(pt[1])
    und = undistort_points([pt], K, dist)[0]
    return float(und[0]), float(und[1])


def load_intrinsics_from_calib(calib: dict, frame_w: int = 1920, frame_h: int = 1080):
    """Return (K, dist) or (None, None) if not configured."""
    if "dist_coeffs" not in calib and "k1" not in calib:
        return None, None
    if "camera_matrix" in calib:
        K = np.array(calib["camera_matrix"], dtype=np.float64)
    else:
        fov = float(calib.get("fov_deg", 85.0))
        K = default_camera_matrix(frame_w, frame_h, fov)
    if "dist_coeffs" in calib:
        dist = np.array(calib["dist_coeffs"], dtype=np.float64).ravel()
    else:
        dist = dist_from_k1(float(calib["k1"]))
    return K, dist
