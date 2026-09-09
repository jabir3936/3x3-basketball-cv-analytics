import cv2
import json
import numpy as np

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

VIDEO_PATH = "media/footage/court_footage.mp4"
TRACKING_JSON = "data/tracking/latest/match_tracking_data_teams.json"
FSM_JSON = "data/fsm/match_fsm_data.json"
OUTPUT_VIDEO_PATH = "media/renders/match_video_diagnostic_with_map.mp4"

# Canvas dimensions (square court map)
CANVAS_W = 720
CANVAS_H = 720
MARGIN = 40

# Court mapping
SCALE = 40.0
MIN_X = -8.0
MIN_Y = -2.0

# Calibrated hoop & arc
HOOP_COURT_COORD = [0.904, 8.036]
ARC_RADIUS_M = 6.248

# Map colors (BGR)
TEAM_COLORS = {
    "team_blue": (255, 130, 0),
    "team_white": (0, 0, 230),
    None: (160, 160, 160)
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
    py = int(CANVAS_H - MARGIN - (y - MIN_Y) * SCALE)
    return (px, py)

def draw_static_court(canvas):
    line_color = (80, 80, 80)

    # Half-court boundary
    cv2.rectangle(canvas, court_to_canvas(-7.5, 0), court_to_canvas(7.5, 11), line_color, 2)

    # Paint / key
    cv2.rectangle(canvas, court_to_canvas(-2.45, 0), court_to_canvas(2.45, 5.8), line_color, 2)

    # 3-point arc + hoop
    hoop_px = court_to_canvas(HOOP_COURT_COORD[0], HOOP_COURT_COORD[1])
    radius_px = int(ARC_RADIUS_M * SCALE)
    cv2.circle(canvas, hoop_px, radius_px, (100, 100, 255), 2)
    cv2.circle(canvas, hoop_px, 6, (0, 0, 255), -1)

def draw_text_bg(canvas, text, pos, color=(255, 255, 255)):
    x, y = pos
    cv2.rectangle(canvas, (x - 2, y - 16), (x + 280, y + 6), (20, 20, 20), -1)
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
# VIDEO SETUP
# ---------------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

aspect_ratio = orig_w / orig_h
resized_w = int(CANVAS_H * aspect_ratio)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (resized_w + CANVAS_W, CANVAS_H))

# Trajectory tail memory (last 15 points per player)
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
    # LEFT SIDE: DIAGNOSTIC OVERLAY (boxes + tails)
    # =====================================================
    if is_valid:
        for obj in track_fr.get("objects", []):
            x1, y1, x2, y2 = map(int, obj["bbox"])
            track_id = obj["track_id"]
            cls_name = obj["class"]

            if cls_name == "player":
                # Unique persistent color per track ID
                color = (
                    (track_id * 37) % 255,
                    (track_id * 113) % 255,
                    (track_id * 211) % 255
                )

                # Movement tail
                center_x, center_y = int((x1 + x2) / 2), y2
                hist = trajectory_history.setdefault(track_id, [])
                hist.append((center_x, center_y))
                if len(hist) > 15:
                    hist.pop(0)

                for i in range(1, len(hist)):
                    cv2.line(frame, hist[i - 1], hist[i], color, 2)
            else:
                color = (0, 165, 255)  # Orange for ball

            # Bounding box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID: {track_id}" if cls_name == "player" else "BALL"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 0, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        # Circuit breaker state (cuts / replays / invalid views)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        text = f"CIRCUIT BREAKER: {cut_reason}"
        font_scale = 1.5
        thickness = 4
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.putText(
            frame, text,
            ((frame.shape[1] - text_w) // 2, (frame.shape[0] + text_h) // 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness
        )

        # Clear tails so lines don't jump across cuts
        trajectory_history = {}

    # Frame counter
    cv2.putText(
        frame, f"Frame: {frame_id} / {total_frames}", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )

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

    if is_valid:
        # Players
        for obj in track_fr.get("objects", []):
            if obj.get("class") != "player":
                continue

            court = obj.get("court_coords_meters")
            if not court or obj.get("court_coords_valid") is False:
                continue

            px = court_to_canvas(court[0], court[1])
            color = TEAM_COLORS.get(obj.get("team_id"), (150, 150, 150))

            cv2.circle(canvas, px, 14, color, -1)
            cv2.circle(canvas, px, 14, (255, 255, 255), 1)
            cv2.putText(canvas, str(obj.get("track_id")), (px[0] - 12, px[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            if obj.get("track_id") == poss_id:
                cv2.circle(canvas, px, 20, (0, 255, 255), 3)
                poss_px = px

        # Ball
        ball_obj = next(
            (o for o in track_fr.get("objects", []) if o.get("class") == "ball"),
            None
        )
        if (
            ball_obj is not None and
            ball_obj.get("court_coords_meters") and
            ball_obj.get("court_coords_valid") is not False
        ):
            court = ball_obj["court_coords_meters"]
            ball_px = court_to_canvas(court[0], court[1])
            cv2.circle(canvas, ball_px, 7, BALL_COLORS.get(ball_state, (100, 100, 100)), -1)
            cv2.circle(canvas, ball_px, 7, (255, 255, 255), 1)

        # Possession link
        if poss_px and ball_px:
            cv2.line(canvas, poss_px, ball_px, (0, 255, 255), 2)

    # FSM HUD
    draw_text_bg(canvas, f"Ball State: {ball_state}", (10, 30),
                 BALL_COLORS.get(ball_state, (255, 255, 255)))

    poss_text = f"Possession: ID {poss_id} ({poss_team})" if poss_id else "Possession: NONE"
    draw_text_bg(canvas, poss_text, (10, 60), (0, 255, 255) if poss_id else (200, 200, 200))

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