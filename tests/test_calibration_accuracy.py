"""Calibration accuracy using the same projection path as production (incl. undistort)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.court_projection import load_calibration, px_to_meters

H, K, dist, _ = load_calibration("data/calibration/calibration_official_tuned.json")

test_pixels = [
    ((350, 570), "key-left baseline", (-2.45, 0.0)),
    ((857, 513), "key-right baseline", (2.45, 0.0)),
    ((613, 784), "FT-left", (-2.45, 5.80)),
    ((1172, 685), "FT-right", (2.45, 5.80)),
]

print("=== TEST 1: CALIBRATION ACCURACY ===")
all_ok = True
for (px, py), name, (expected_x, expected_y) in test_pixels:
    mx, my = px_to_meters(H, (px, py), K, dist)
    mx, my = round(mx, 2), round(my, 2)
    err = ((mx - expected_x) ** 2 + (my - expected_y) ** 2) ** 0.5
    status = "PASS" if err < 0.5 else "FAIL"
    if status == "FAIL":
        all_ok = False
    print(
        f"  {name}: pixel ({px},{py}) -> ({mx}, {my})  "
        f"expected ({expected_x}, {expected_y})  err={err:.2f}m  [{status}]"
    )
raise SystemExit(0 if all_ok else 1)
