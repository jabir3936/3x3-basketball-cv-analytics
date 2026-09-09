"""
Short side-by-side preview to judge 2D map accuracy without full-match render.

Usage:
  python scripts/diagnostics/render_mapping_preview.py
  python scripts/diagnostics/render_mapping_preview.py --start 200 --frames 450
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

# Reuse constants / helpers from main renderer by import would pull full script;
# keep a thin copy of court drawing for the preview.

VIDEO = ROOT / "media/footage/court_footage.mp4"
TRACKING = ROOT / "data/tracking/latest/match_tracking_data_smooth.json"
FSM = ROOT / "data/fsm/match_fsm_data.json"
OUT = ROOT / "media/renders/mapping_preview_clip.mp4"

CANVAS_W = CANVAS_H = 720
MARGIN = 40
SCALE = 40.0
MIN_X, MIN_Y = -8.0, -1.0
TEAM_COLORS = {"team_blue": (255, 130, 0), "team_white": (0, 0, 230)}


def court_to_canvas(x, y):
    return (int(MARGIN + (x - MIN_X) * SCALE), int(MARGIN + (y - MIN_Y) * SCALE))


def draw_static_court(canvas):
    L = (80, 80, 80)
    for a, b in [((-7.5, 0), (7.5, 0)), ((7.5, 0), (7.5, 11)),
                 ((7.5, 11), (-7.5, 11)), ((-7.5, 11), (-7.5, 0))]:
        cv2.line(canvas, court_to_canvas(*a), court_to_canvas(*b), L, 2)
    for a, b in [((-2.45, 0), (2.45, 0)), ((2.45, 0), (2.45, 5.8)),
                 ((2.45, 5.8), (-2.45, 5.8)), ((-2.45, 5.8), (-2.45, 0))]:
        cv2.line(canvas, court_to_canvas(*a), court_to_canvas(*b), L, 2)
    cv2.circle(canvas, court_to_canvas(0, 1.575), 6, (0, 0, 255), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=300)
    ap.add_argument("--frames", type=int, default=400)
    args = ap.parse_args()

    with open(TRACKING) as f:
        tracking = {fr["frame_id"]: fr for fr in json.load(f)["frames"]}
    fsm_frames = {}
    if FSM.exists():
        with open(FSM) as f:
            fsm_frames = {fr["frame_id"]: fr for fr in json.load(f)["frames"]}

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resized_w = int(CANVAS_H * (orig_w / orig_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start - 1)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(OUT), fourcc, fps, (resized_w + CANVAS_W, CANVAS_H))

    for i in range(args.frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_id = args.start + i
        track_fr = tracking.get(frame_id, {})
        fsm_info = fsm_frames.get(frame_id, {}).get("fsm", {})
        poss_id = fsm_info.get("possession_player_id")
        ball_state = fsm_info.get("ball_state", "BALL_UNKNOWN")

        # left: feet overlay
        for obj in track_fr.get("objects", []):
            if obj.get("class") != "player" or not obj.get("bbox"):
                continue
            x1, y1, x2, y2 = map(int, obj["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            feet = obj.get("feet_coord_px")
            if feet:
                cv2.circle(frame, (int(feet[0]), int(feet[1])), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"frame {frame_id}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        left = cv2.resize(frame, (resized_w, CANVAS_H))

        # right: map
        canvas = 35 * np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        draw_static_court(canvas)
        player_by_id = {}
        for obj in track_fr.get("objects", []):
            if obj.get("class") != "player":
                continue
            team = obj.get("team_id")
            if team not in TEAM_COLORS:
                continue
            court = obj.get("court_coords_meters")
            if not court or obj.get("court_coords_valid") is False:
                continue
            px = court_to_canvas(court[0], court[1])
            player_by_id[obj.get("track_id")] = court
            cv2.circle(canvas, px, 14, TEAM_COLORS[team], -1)
            if obj.get("track_id") == poss_id:
                cv2.circle(canvas, px, 20, (0, 255, 255), 3)

        ball_draw = None
        if poss_id is not None and poss_id in player_by_id:
            ball_draw = player_by_id[poss_id]
        else:
            ball_obj = next((o for o in track_fr.get("objects", []) if o.get("class") == "ball"), None)
            if ball_obj and ball_obj.get("court_coords_meters") and ball_obj.get("court_coords_valid") is not False:
                ball_draw = ball_obj["court_coords_meters"]
        if ball_draw is not None:
            bp = court_to_canvas(ball_draw[0], ball_draw[1])
            cv2.circle(canvas, bp, 7, (0, 220, 0), -1)
            cv2.circle(canvas, bp, 7, (255, 255, 255), 1)

        cv2.putText(canvas, f"{ball_state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        out.write(np.hstack((left, canvas)))

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{args.frames}")

    cap.release()
    out.release()
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
