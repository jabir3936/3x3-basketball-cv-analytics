import json
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
FRAME_W, FRAME_H = 1920, 1080

H2, K, dist, _ = load_calibration(CALIBRATION_FILE)

with open(INPUT_JSON, "r") as f:
    payload = json.load(f)

xs, ys = [], []
converted = 0
invalid = 0

for frame in payload.get("frames", []):
    for obj in frame.get("objects", []):
        if obj.get("class") == "player":
            bbox = obj.get("bbox")
            if not bbox:
                obj["court_coords_meters"] = [0.0, 0.0]
                obj["court_coords_valid"] = False
                invalid += 1
                continue
            feet, court, valid = project_player_feet(
                H2, bbox, FRAME_W, FRAME_H, K, dist
            )
            if feet is not None:
                obj["feet_coord_px"] = feet
            if court is None:
                obj["court_coords_meters"] = [0.0, 0.0]
                obj["court_coords_valid"] = False
                invalid += 1
                continue
            obj["court_coords_meters"] = court
            obj["court_coords_valid"] = bool(valid)
            if valid:
                converted += 1
                xs.append(court[0])
                ys.append(court[1])
            else:
                invalid += 1
        else:
            pix = obj.get("center_coord_px")
            if pix is None:
                obj["court_coords_meters"] = [0.0, 0.0]
                obj["court_coords_valid"] = False
                invalid += 1
                continue
            court, ok = project_ball_center(H2, pix, K, dist)
            obj["court_coords_meters"] = court
            obj["court_coords_valid"] = bool(ok)
            if ok:
                converted += 1
            else:
                invalid += 1

with open(OUTPUT_JSON, "w") as f:
    json.dump(payload, f, indent=2)

print(f"Saved: {OUTPUT_JSON}")
print(f"Converted objects: {converted} | invalid: {invalid}")
if xs:
    print(f"Player X range: {min(xs):.2f} .. {max(xs):.2f}  (expect about -7.5 .. 7.5)")
    print(f"Player Y range: {min(ys):.2f} .. {max(ys):.2f}  (expect about 0 .. 11)")
