"""
Write diagnostic stills: bbox + feet dots on video frames.

Usage:
  python scripts/diagnostics/overlay_feet_check.py
  python scripts/diagnostics/overlay_feet_check.py --frames 100,500,1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "media/footage/court_footage.mp4"
TRACKING = ROOT / "data/tracking/latest/match_tracking_data_official.json"
OUT_DIR = ROOT / "media/checks"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="100,400,800,1200,2000", help="comma-separated frame_ids")
    ap.add_argument("--json", default=str(TRACKING))
    args = ap.parse_args()
    frame_ids = [int(x) for x in args.frames.split(",") if x.strip()]

    with open(args.json) as f:
        frames = {fr["frame_id"]: fr for fr in json.load(f)["frames"]}

    cap = cv2.VideoCapture(str(VIDEO))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid - 1)
        ret, frame = cap.read()
        if not ret:
            print(f"skip frame {fid}: read failed")
            continue
        fr = frames.get(fid, {})
        for obj in fr.get("objects", []):
            if obj.get("class") != "player" or not obj.get("bbox"):
                continue
            x1, y1, x2, y2 = map(int, obj["bbox"])
            valid = obj.get("court_coords_valid", True)
            color = (0, 255, 0) if valid else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            feet = obj.get("feet_coord_px")
            if feet:
                cv2.circle(frame, (int(feet[0]), int(feet[1])), 6, (0, 0, 255), -1)
                cv2.circle(frame, (int(feet[0]), int(feet[1])), 8, (255, 255, 255), 1)
            # also mark raw bbox bottom-center for comparison (cyan)
            cx, cy = int((x1 + x2) / 2), y2
            cv2.circle(frame, (cx, cy), 4, (255, 255, 0), -1)
            court = obj.get("court_coords_meters")
            label = f"id{obj.get('track_id')}"
            if court:
                label += f" ({court[0]:.1f},{court[1]:.1f}m)"
            cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        out = OUT_DIR / f"feet_check_frame_{fid:04d}.jpg"
        cv2.imwrite(str(out), frame)
        print(f"Saved {out}")

    cap.release()
    print("Red = feet_coord_px | Cyan = raw bbox bottom | Green box = valid, Red box = invalid")


if __name__ == "__main__":
    main()
