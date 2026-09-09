"""
Recompute player feet from bbox, re-project court meters with current H.

Usage (from project root):
  python scripts/03_processing/recompute_feet_and_project.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.court_projection import (  # noqa: E402
    load_calibration,
    project_ball_center,
    project_player_feet,
)

CALIBRATION_FILE = ROOT / "data/calibration/calibration_official_tuned.json"
INPUT_JSON = ROOT / "data/tracking/latest/match_tracking_data_teams_new_7Sep.json"
OUTPUT_JSON = ROOT / "data/tracking/latest/match_tracking_data_official.json"

# Match video resolution (court_footage.mp4)
FRAME_W, FRAME_H = 1920, 1080


def main():
    H, K, dist, _ = load_calibration(CALIBRATION_FILE)
    with open(INPUT_JSON) as f:
        payload = json.load(f)

    xs, ys = [], []
    n_ok = n_bad_feet = n_bad_proj = 0

    for frame in payload.get("frames", []):
        for obj in frame.get("objects", []):
            cls = obj.get("class")
            if cls == "player":
                bbox = obj.get("bbox")
                if not bbox:
                    obj["feet_coord_px"] = None
                    obj["court_coords_meters"] = [0.0, 0.0]
                    obj["court_coords_valid"] = False
                    n_bad_feet += 1
                    continue
                feet, court, valid = project_player_feet(
                    H, bbox, FRAME_W, FRAME_H, K, dist
                )
                if feet is None:
                    x1, y1, x2, y2 = map(float, bbox)
                    obj["feet_coord_px"] = [int((x1 + x2) / 2), int(y2)]
                    obj["court_coords_meters"] = [0.0, 0.0]
                    obj["court_coords_valid"] = False
                    obj["feet_reject_reason"] = "edge_clipped_or_bad_bbox"
                    n_bad_feet += 1
                    continue
                obj["feet_coord_px"] = feet
                obj.pop("feet_reject_reason", None)
                if court is None:
                    obj["court_coords_meters"] = [0.0, 0.0]
                    obj["court_coords_valid"] = False
                    n_bad_proj += 1
                    continue
                obj["court_coords_meters"] = court
                obj["court_coords_valid"] = bool(valid)
                if valid:
                    n_ok += 1
                    xs.append(court[0])
                    ys.append(court[1])
                else:
                    n_bad_proj += 1
            elif cls == "ball":
                pix = obj.get("center_coord_px")
                if pix is None and obj.get("bbox"):
                    x1, y1, x2, y2 = obj["bbox"]
                    pix = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                    obj["center_coord_px"] = pix
                if pix is None:
                    obj["court_coords_meters"] = [0.0, 0.0]
                    obj["court_coords_valid"] = False
                    n_bad_proj += 1
                    continue
                court, ok = project_ball_center(H, pix, K, dist)
                obj["court_coords_meters"] = court
                obj["court_coords_valid"] = bool(ok and all(math.isfinite(v) for v in court))
                if obj["court_coords_valid"]:
                    n_ok += 1
                else:
                    n_bad_proj += 1

    with open(OUTPUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved: {OUTPUT_JSON}")
    print(f"OK projections: {n_ok} | bad feet: {n_bad_feet} | bad/out-of-bounds: {n_bad_proj}")
    if xs:
        print(f"Player X range: {min(xs):.2f} .. {max(xs):.2f}  (expect ~ -7.5 .. 7.5)")
        print(f"Player Y range: {min(ys):.2f} .. {max(ys):.2f}  (expect ~ 0 .. 11)")


if __name__ == "__main__":
    main()
