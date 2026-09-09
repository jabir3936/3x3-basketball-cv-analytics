import cv2
import json
import math
import numpy as np

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

VIDEO_PATH = "media/footage/court_footage.mp4"
TRACKING_JSON = "data/tracking/latest/match_tracking_data_smooth.json"
FSM_JSON = "data/fsm/match_fsm_data.json"
OUTPUT_VIDEO_PATH = "media/renders/match_video_diagnostic_with_map.mp4"

CANVAS_W = 720
CANVAS_H = 720
MARGIN = 40

# official metric mapping, baseline at TOP like the video
SCALE = 40.0
MIN_X = -8.0
MIN_Y = -1.0

# Playback speed of the OUTPUT video (1.0 = real time, 0.5 = half, 0.25 = quarter)
PLAYBACK_SPEED = 0.75

TEAM_COLORS = {
    "team_blue": (255, 130, 0),   # blue dot
    "team_white": (0, 0, 230),    # red dot
}

BALL_COLORS = {
    "BALL_DETECTED": (0, 220, 0),
    "BALL_INTERPOLATED": (0, 220, 220),
    "BALL_OCCLUDED": (0, 140, 255),
    "BALL_LOST": (0, 0, 255),
    "BALL_UNKNOWN": (120, 120, 120)
}

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def court_to_canvas(x, y):
    px = int(MARGIN + (x - MIN_X) * SCALE)
    py = int(MARGIN + (y - MIN_Y) * SCALE)
    return (px, py)

def draw_static_court(canvas):
    L = (80, 80, 80)
    # 15 x 11 boundary
    for a, b in [((-7.5,0),(7.5,0)), ((7.5,0),(7.5,11)),
                 ((7.5,11),(-7.5,11)), ((-7.5,11),(-7.5,0))]:
        cv2.line(canvas, court_to_canvas(*a), court_to_canvas(*b), L, 2)
    # 4.90 x 5.80 key
    for a, b in [((-2.45,0),(2.45,0)), ((2.45,0),(2.45,5.8)),
                 ((2.45,5.8),(-2.45,5.8)), ((-2.45,5.8),(-2.45,0))]:
        cv2.line(canvas, court_to_canvas(*a), court_to_canvas(*b), L, 2)
    # arc straight sections (0.90 m from sidelines)
    cv2.line(canvas, court_to_canvas(-6.6,0), court_to_canvas(-6.6,2.99), L, 2)
    cv2.line(canvas, court_to_canvas(6.6,0),  court_to_canvas(6.6,2.99),  L, 2)
    # true 6.75 m arc centered at hoop (0, 1.575)
    prev = None
    for deg in range(0, 181, 3):
        t = math.radians(deg)
        p = court_to_canvas(6.75*math.cos(t), 1.575 + 6.75*math.sin(t))
        if prev:
            cv2.line(canvas, prev, p, (100,100,255), 2)
        prev = p
    # hoop
    cv2.circle(canvas, court_to_canvas(0, 1.575), 6, (0,0,255), -1)

def draw_text_bg(canvas, text, pos, color=(255,255,255)):
    x, y = pos
    cv2.rectangle(canvas, (x-2, y-16), (x+280, y+6), (20,20,20), -1)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

# ---------------------------------------------------------
# LOAD TELEMETRY
# ---------------------------------------------------------

print("Loading JSON telemetry...")
with open(TRACKING_JSON, "r") as f:
    tracking_frames = {fr["frame_id"]: fr for fr in json.load(f)["frames"]}

with open(FSM_JSON, "r") as f:
    fsm_frames = {fr["frame_id"]: fr for fr in json.load(f)["frames"]}

# ---------------------------------------------------------
# VIDEO SETUP (SLOW MOTION)
# ---------------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

aspect_ratio = orig_w / orig_h
resized_w = int(CANVAS_H * aspect_ratio)

out_fps = max(1.0, fps * PLAYBACK_SPEED)
print(f"Writing output at {out_fps:.2f} fps ({PLAYBACK_SPEED}x speed)")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, out_fps, (resized_w + CANVAS_W, CANVAS_H))

trajectory_history = {}

print(f"Rendering {total_frames} frames...")
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    frame_id = frame_idx

    track_fr = tracking_frames.get(frame_id, {})
    fsm_info = fsm_frames.get(frame_id, {}).get("fsm", {})

    is_valid = track_fr.get("is_valid_court_view", True)
    cut_reason = track_fr.get("cut_reason", "")

    # =====================================================
    # LEFT SIDE: DIAGNOSTIC OVERLAY (unchanged)
    # =====================================================
    if is_valid:
        for obj in track_fr.get("objects", []):
            x1, y1, x2, y2 = map(int, obj["bbox"])
            track_id = obj["track_id"]
            cls_name = obj["class"]

            if cls_name == "player":
                color = (
                    (track_id * 37) % 255,
                    (track_id * 113) % 255,
                    (track_id * 211) % 255
                )
                center_x, center_y = int((x1 + x2) / 2), y2
                hist = trajectory_history.setdefault(track_id, [])
                hist.append((center_x, center_y))
                if len(hist) > 15:
                    hist.pop(0)
                for i in range(1, len(hist)):
                    cv2.line(frame, hist[i - 1], hist[i], color, 2)
            else:
                color = (0, 165, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID: {track_id}" if cls_name == "player" else "BALL"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 0, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        text = f"CIRCUIT BREAKER: {cut_reason}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
        cv2.putText(frame, text,
                    ((frame.shape[1] - text_w) // 2, (frame.shape[0] + text_h) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        trajectory_history = {}

    cv2.putText(frame, f"Frame: {frame_id} / {total_frames}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    frame_resized = cv2.resize(frame, (resized_w, CANVAS_H))

    # =====================================================
    # RIGHT SIDE: 2D COURT MAP + FSM HUD
    # =====================================================
    canvas = 35 * np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    draw_static_court(canvas)

    poss_id = fsm_info.get("possession_player_id")
    poss_team = fsm_info.get("possession_team_id")
    ball_state = fsm_info.get("ball_state", "BALL_UNKNOWN")
    clearance = fsm_info.get("clearance_state", "NOT_REQUIRED")
    zone = fsm_info.get("ball_zone", "UNKNOWN")

    poss_px = None
    ball_px = None
    player_by_id = {}

    if is_valid:
        # Players: ONLY blue / red dots, no numbers, no gray
        for obj in track_fr.get("objects", []):
            if obj.get("class") != "player":
                continue
            team = obj.get("team_id")
            if team not in TEAM_COLORS:      # skip referees / unlabeled
                continue
            court = obj.get("court_coords_meters")
            if not court or obj.get("court_coords_valid") is False:
                continue

            px = court_to_canvas(court[0], court[1])
            player_by_id[obj.get("track_id")] = (px, court)
            cv2.circle(canvas, px, 14, TEAM_COLORS[team], -1)   # plain team dot

            if obj.get("track_id") == poss_id:
                cv2.circle(canvas, px, 20, (0, 255, 255), 3)    # possession ring
                poss_px = px

        # Ball on map: use ground proxy when possessed (airborne center through H is wrong).
        # - POSSESSED: snap to possession player's court feet
        # - LOOSE / DETECTED without possession: project center, but fade if far from all players
        # - LOST / UNKNOWN: hide or keep last only if we have a valid ground sample
        ball_obj = next((o for o in track_fr.get("objects", []) if o.get("class") == "ball"), None)
        ball_draw = None
        ball_proxy = False

        if poss_id is not None and poss_id in player_by_id:
            ball_draw = player_by_id[poss_id][1]
            ball_proxy = True
        elif (
            ball_obj
            and ball_obj.get("court_coords_meters")
            and ball_obj.get("court_coords_valid") is not False
            and ball_state in ("BALL_DETECTED", "BALL_INTERPOLATED", "BALL_OCCLUDED")
        ):
            # Loose ball: only show raw projection if near some player (likely near floor)
            raw = ball_obj["court_coords_meters"]
            near = False
            for _, court in player_by_id.values():
                if math.hypot(raw[0] - court[0], raw[1] - court[1]) < 3.5:
                    near = True
                    break
            if near or not player_by_id:
                ball_draw = raw

        if ball_draw is not None:
            ball_px = court_to_canvas(ball_draw[0], ball_draw[1])
            color = BALL_COLORS.get(ball_state, (100, 100, 100))
            cv2.circle(canvas, ball_px, 7, color, -1)
            cv2.circle(canvas, ball_px, 7, (255, 255, 255), 1)
            if ball_proxy:
                cv2.putText(canvas, "ball:ground", (ball_px[0] + 10, ball_px[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        # Possession link (only when ball not snapped onto the same player)
        if poss_px and ball_px and not ball_proxy:
            cv2.line(canvas, poss_px, ball_px, (0, 255, 255), 2)

    # HUD
    draw_text_bg(canvas, f"Ball State: {ball_state}", (10, 30),
                 BALL_COLORS.get(ball_state, (255, 255, 255)))
    poss_text = f"Possession: {poss_team}" if poss_team else "Possession: NONE"
    draw_text_bg(canvas, poss_text, (10, 60), (0, 255, 255) if poss_team else (200, 200, 200))
    draw_text_bg(canvas, f"Shot Zone: {zone}", (10, 90))
    clear_color = (0, 255, 0) if clearance == "CLEARANCE_COMPLETED" else (255, 255, 255)
    draw_text_bg(canvas, f"Clearance: {clearance}", (10, 120), clear_color)

    # =====================================================
    # COMBINE + WRITE
    # =====================================================
    combined = np.hstack((frame_resized, canvas))
    out.write(combined)

    if frame_idx % 300 == 0:
        print(f"Processed {frame_idx}/{total_frames} frames...")

cap.release()
out.release()
print(f"Done! Saved: {OUTPUT_VIDEO_PATH}")