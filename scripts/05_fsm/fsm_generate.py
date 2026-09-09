import json
import math


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

INPUT_JSON = "data/tracking/archive/match_tracking_data_teams_new_1Sep.json"
OUTPUT_JSON = "data/fsm/match_fsm_data_newsep.json"

FPS = 30.0

# Calibrated hoop / arc for THIS camera coordinate frame
HOOP_COURT_COORD = [0.0, 1.575]
ARC_RADIUS_M = 6.75
ARC_HYSTERESIS_M = 0.15

# Possession rules (updated for dribbling + fragmented detections)
POSSESSION_RADIUS_M = 1.2
LOOSE_BALL_DISTANCE_M = 2.0
POSSESSION_CONFIRM_FRAMES = 4
POSSESSION_RELEASE_FRAMES = 15

# NEW: allow short ball dropouts while confirming possession
CANDIDATE_GAP_MAX_FRAMES = 3

# ID-switch continuity (track_id changes)
ID_SWITCH_RADIUS_M = 1.0
ID_SWITCH_MAX_FRAMES = 8

TEAMS = ["team_blue", "team_white"]


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def get_court_coord(obj):
    if obj is None:
        return None

    if obj.get("court_coords_valid") is False:
        return None

    coord = obj.get("court_coords_meters")
    if coord is None or len(coord) != 2:
        return None

    try:
        return [float(coord[0]), float(coord[1])]
    except Exception:
        return None


def distance_m(a, b):
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def select_best_ball(objects):
    balls = [o for o in objects if o.get("class") == "ball"]
    if not balls:
        return None

    def score(b):
        s = float(b.get("confidence", 0.0))
        if bool(b.get("detected", True)):
            s += 0.5
        if bool(b.get("is_interpolated", False)):
            s -= 0.3
        if get_court_coord(b) is not None:
            s += 0.1
        return s

    return max(balls, key=score)


def other_team(team_id):
    if team_id == "team_blue":
        return "team_white"
    if team_id == "team_white":
        return "team_blue"
    return None


# ---------------------------------------------------------
# RUN FSM
# ---------------------------------------------------------

def run_fsm(payload):
    frames_out = []
    events = []

    # FSM memory
    ball_present_prev = False

    possession_state = "NO_POSSESSION"
    possession_player_id = None
    possession_team_id = None
    possession_court = None

    candidate_id = None
    candidate_count = 0
    candidate_gap = 0
    release_count = 0
    missing_count = 0

    clearance_state = "CLEARANCE_NOT_REQUIRED"

    def add_event(frame_id, event_type, **extra):
        evt = {
            "event_id": f"evt_{len(events) + 1:04d}",
            "frame_id": frame_id,
            "timestamp_seconds": round((frame_id - 1) / FPS, 3),
            "event_type": event_type,
        }
        evt.update(extra)
        events.append(evt)
        return event_type

    def reset_clearance_on_gain(frame_id, ref_coord):
        nonlocal clearance_state
        if ref_coord is None:
            clearance_state = "CLEARANCE_NOT_REQUIRED"
            return

        if distance_m(ref_coord, HOOP_COURT_COORD) < ARC_RADIUS_M:
            clearance_state = "CLEARANCE_REQUIRED"
            add_event(frame_id, "CLEARANCE_REQUIRED", team_id=possession_team_id)
        else:
            clearance_state = "CLEARANCE_NOT_REQUIRED"

    for frame in payload.get("frames", []):
        frame_id = frame.get("frame_id")
        is_valid = bool(frame.get("is_valid_court_view", True))
        objects = frame.get("objects", [])

        frame_events = []

        # -------------------------------------------------
        # Invalid frame: hold state, no decisions
        # -------------------------------------------------
        if not is_valid:
            frames_out.append({
                "frame_id": frame_id,
                "is_valid_court_view": False,
                "fsm": {
                    "ball_state": "BALL_UNKNOWN",
                    "possession_state": possession_state,
                    "possession_player_id": possession_player_id,
                    "possession_team_id": possession_team_id,
                    "defending_team_id": other_team(possession_team_id),
                    "clearance_state": clearance_state,
                    "ball_zone": None,
                    "events": [],
                },
            })
            continue

        # -------------------------------------------------
        # BALL STATE
        # -------------------------------------------------
        ball = select_best_ball(objects)
        ball_court = get_court_coord(ball) if ball else None

        if ball is None:
            ball_state = "BALL_LOST"
        elif ball.get("is_interpolated", False):
            ball_state = "BALL_INTERPOLATED"
        elif ball.get("is_occluded", False):
            ball_state = "BALL_OCCLUDED"
        else:
            ball_state = "BALL_DETECTED"

        ball_present = ball is not None

        if ball_present and not ball_present_prev:
            frame_events.append(add_event(frame_id, "BALL_RECOVERED"))
        if not ball_present and ball_present_prev:
            frame_events.append(add_event(frame_id, "BALL_LOST_EVENT"))
        ball_present_prev = ball_present

        # -------------------------------------------------
        # PLAYERS + NEAREST TO BALL
        # -------------------------------------------------
        players = []
        for obj in objects:
            if obj.get("class") != "player":
                continue
            court = get_court_coord(obj)
            if court is None:
                continue
            players.append({
                "track_id": obj.get("track_id"),
                "team_id": obj.get("team_id"),
                "court": court,
            })

        nearest_id = None
        nearest_team = None
        nearest_court = None
        nearest_dist = None

        if ball_court is not None:
            best = float("inf")
            for p in players:
                d = distance_m(ball_court, p["court"])
                if d < best:
                    best = d
                    nearest_id = p["track_id"]
                    nearest_team = p["team_id"]
                    nearest_court = p["court"]
            if nearest_id is not None:
                nearest_dist = best

        candidate = (
            nearest_id
            if nearest_dist is not None and nearest_dist <= POSSESSION_RADIUS_M
            else None
        )

        # -------------------------------------------------
        # POSSESSION FSM
        # -------------------------------------------------
        if possession_state == "PLAYER_POSSESSION":
            possessor = next(
                (p for p in players if p["track_id"] == possession_player_id),
                None
            )

            if candidate == possession_player_id:
                # Possessor still controls the ball
                release_count = 0
                missing_count = 0
                candidate_count = 0
                candidate_gap = 0
                possession_court = nearest_court

            elif candidate is not None:
                # A different player is closest to the ball
                release_count = 0
                candidate_gap = 0

                if candidate_id == candidate:
                    candidate_count += 1
                else:
                    candidate_id = candidate
                    candidate_count = 1

                if candidate_count >= POSSESSION_CONFIRM_FRAMES:
                    old_id = possession_player_id
                    possession_player_id = candidate
                    possession_team_id = nearest_team
                    possession_court = nearest_court
                    candidate_count = 0
                    candidate_gap = 0

                    frame_events.append(add_event(
                        frame_id,
                        "POSSESSION_CHANGE",
                        from_player_id=old_id,
                        to_player_id=possession_player_id,
                        team_id=possession_team_id
                    ))
                    reset_clearance_on_gain(frame_id, possession_court)

            else:
                # No candidate near the ball
                candidate_gap += 1
                if candidate_gap > CANDIDATE_GAP_MAX_FRAMES:
                    candidate_count = 0
                    candidate_id = None

                if possessor is not None:
                    # Possessor still visible on court
                    if (
                        ball_present and
                        nearest_dist is not None and
                        nearest_dist > LOOSE_BALL_DISTANCE_M
                    ):
                        # Ball is away from everybody -> loose
                        frame_events.append(add_event(
                            frame_id,
                            "POSSESSION_LOSS",
                            player_id=possession_player_id,
                            reason="LOOSE_BALL"
                        ))
                        possession_state = "LOOSE_BALL"
                        possession_player_id = None
                        possession_team_id = None
                        possession_court = None
                        clearance_state = "CLEARANCE_NOT_REQUIRED"
                    else:
                        release_count += 1
                        if release_count > POSSESSION_RELEASE_FRAMES:
                            frame_events.append(add_event(
                                frame_id,
                                "POSSESSION_LOSS",
                                player_id=possession_player_id,
                                reason="RELEASED"
                            ))
                            possession_state = "NO_POSSESSION"
                            possession_player_id = None
                            possession_team_id = None
                            possession_court = None
                            clearance_state = "CLEARANCE_NOT_REQUIRED"
                else:
                    # Possessor track_id disappeared -> try ID-switch transfer
                    transferred = False

                    if (
                        possession_court is not None and
                        missing_count <= ID_SWITCH_MAX_FRAMES
                    ):
                        best_p = None
                        best_d = float("inf")
                        for p in players:
                            d = distance_m(possession_court, p["court"])
                            if d < best_d:
                                best_d = d
                                best_p = p

                        if best_p is not None and best_d <= ID_SWITCH_RADIUS_M:
                            possession_player_id = best_p["track_id"]
                            if best_p["team_id"] is not None:
                                possession_team_id = best_p["team_id"]
                            possession_court = best_p["court"]
                            transferred = True

                    if not transferred:
                        missing_count += 1
                        release_count += 1

                        if release_count > POSSESSION_RELEASE_FRAMES:
                            frame_events.append(add_event(
                                frame_id,
                                "POSSESSION_LOSS",
                                player_id=possession_player_id,
                                reason="TRACK_LOST"
                            ))
                            possession_state = "NO_POSSESSION"
                            possession_player_id = None
                            possession_team_id = None
                            possession_court = None
                            clearance_state = "CLEARANCE_NOT_REQUIRED"
                    else:
                        missing_count = 0

        else:
            # NO_POSSESSION or LOOSE_BALL
            if candidate is not None:
                if candidate_id == candidate:
                    candidate_count += 1
                else:
                    candidate_id = candidate
                    candidate_count = 1
                candidate_gap = 0

                if candidate_count >= POSSESSION_CONFIRM_FRAMES:
                    possession_state = "PLAYER_POSSESSION"
                    possession_player_id = candidate
                    possession_team_id = nearest_team
                    possession_court = nearest_court
                    candidate_count = 0
                    candidate_gap = 0
                    release_count = 0

                    frame_events.append(add_event(
                        frame_id,
                        "POSSESSION_GAIN",
                        player_id=possession_player_id,
                        team_id=possession_team_id
                    ))
                    reset_clearance_on_gain(frame_id, possession_court)
            else:
                # Tolerate short ball dropouts while confirming.
                candidate_gap += 1
                if candidate_gap > CANDIDATE_GAP_MAX_FRAMES:
                    candidate_count = 0
                    candidate_id = None

                if (
                    ball_present and
                    nearest_dist is not None and
                    nearest_dist > LOOSE_BALL_DISTANCE_M
                ):
                    if possession_state != "LOOSE_BALL":
                        possession_state = "LOOSE_BALL"
                        frame_events.append(add_event(frame_id, "LOOSE_BALL"))
                elif not ball_present:
                    possession_state = "NO_POSSESSION"

        # -------------------------------------------------
        # CLEARANCE FSM (3x3 arc rule)
        # -------------------------------------------------
        if (
            clearance_state == "CLEARANCE_REQUIRED" and
            possession_state == "PLAYER_POSSESSION"
        ):
            ref = possession_court if possession_court is not None else ball_court

            if ref is not None and (
                distance_m(ref, HOOP_COURT_COORD) >= ARC_RADIUS_M + ARC_HYSTERESIS_M
            ):
                clearance_state = "CLEARANCE_COMPLETED"
                frame_events.append(add_event(
                    frame_id,
                    "CLEARANCE_COMPLETED",
                    team_id=possession_team_id
                ))

        # -------------------------------------------------
        # BALL ZONE
        # -------------------------------------------------
        ball_zone = None
        if ball_court is not None:
            ball_zone = (
                "INSIDE_ARC"
                if distance_m(ball_court, HOOP_COURT_COORD) < ARC_RADIUS_M
                else "OUTSIDE_ARC"
            )

        # -------------------------------------------------
        # STORE FRAME
        # -------------------------------------------------
        frames_out.append({
            "frame_id": frame_id,
            "is_valid_court_view": True,
            "fsm": {
                "ball_state": ball_state,
                "possession_state": possession_state,
                "possession_player_id": possession_player_id,
                "possession_team_id": possession_team_id,
                "defending_team_id": other_team(possession_team_id),
                "nearest_player_id": nearest_id,
                "nearest_player_distance_m": (
                    round(nearest_dist, 3) if nearest_dist is not None else None
                ),
                "clearance_state": clearance_state,
                "ball_zone": ball_zone,
                "events": frame_events,
            },
        })

    return frames_out, events


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

with open(INPUT_JSON, "r") as f:
    payload = json.load(f)

frames_out, events = run_fsm(payload)

# Summary statistics
event_counts = {}
for e in events:
    event_counts[e["event_type"]] = event_counts.get(e["event_type"], 0) + 1

possession_frames = {"team_blue": 0, "team_white": 0, "unknown": 0}
for fr in frames_out:
    if fr["fsm"]["possession_state"] == "PLAYER_POSSESSION":
        team = fr["fsm"].get("possession_team_id")
        if team in possession_frames:
            possession_frames[team] += 1
        else:
            possession_frames["unknown"] += 1

output_payload = {
    "metadata": {
        "source_json": INPUT_JSON,
        "fps": FPS,
        "hoop_court_coord": HOOP_COURT_COORD,
        "arc_radius_m": ARC_RADIUS_M,
        "possession_radius_m": POSSESSION_RADIUS_M,
        "event_counts": event_counts,
        "possession_frames_by_team": possession_frames,
    },
    "events": events,
    "frames": frames_out,
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(output_payload, f, indent=2)

print("FSM complete.")
print(f"Saved: {OUTPUT_JSON}")
print("\nEvent counts:")
print(json.dumps(event_counts, indent=2))
print("\nPossession frames by team:")
print(json.dumps(possession_frames, indent=2))