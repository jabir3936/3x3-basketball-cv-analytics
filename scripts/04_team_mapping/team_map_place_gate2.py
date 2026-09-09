import json
import csv
import os


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

INPUT_JSON = "data/tracking/archive/match_tracking_data_interpolated_v7.json"
TEAM_CSV = "data/team_mapping/team_mapping_full.csv"

OUTPUT_JSON = "data/tracking/latest/match_tracking_data_teams_new_7Sep.json"

VALID_TEAMS = [
    "team_blue",
    "team_white"
]

# For 3x3 basketball, each team should normally have max 3 players on court.
MAX_PLAYERS_PER_TEAM = 3


# ---------------------------------------------------------
# NORMALIZE TEAM LABELS
# ---------------------------------------------------------

def normalize_team_label(team_label):
    """
    Accept common variations like:
        blue, Blue, team_blue, TEAM_BLUE
        white, White, team_white, TEAM_WHITE
        team white, team-white
    """
    if team_label is None:
        return None

    team_label = str(team_label).strip().lower()

    # Convert spaces and dashes to underscores
    team_label = team_label.replace(" ", "_")
    team_label = team_label.replace("-", "_")

    if team_label == "":
        return None

    if team_label in {"blue", "team_blue", "b"}:
        return "team_blue"

    if team_label in {"white", "team_white", "w"}:
        return "team_white"

    # unknown, empty, invalid labels become None
    return None


# ---------------------------------------------------------
# LOAD TEAM MAPPING CSV
# ---------------------------------------------------------

def load_team_mapping(csv_path):
    """
    Load team mapping from CSV.

    Expected CSV columns:
        segment_id, track_id, scene_block, start_frame, end_frame,
        frame_count, sample_images, team_id
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Team mapping CSV not found: {csv_path}\n"
            f"Please rename your labeled CSV to {csv_path} "
            f"or change TEAM_CSV in SETTINGS."
        )

    mapping = {}

    skipped_rows = 0
    loaded_segments = 0

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            team_id = normalize_team_label(row.get("team_id"))

            # Ignore unknown, empty, or invalid labels
            if team_id is None:
                skipped_rows += 1
                continue

            try:
                track_id = int(str(row.get("track_id")).strip())
                start_frame = int(str(row.get("start_frame")).strip())
                end_frame = int(str(row.get("end_frame")).strip())
            except Exception:
                skipped_rows += 1
                continue

            # If frames are accidentally swapped, fix them
            if end_frame < start_frame:
                start_frame, end_frame = end_frame, start_frame

            if track_id not in mapping:
                mapping[track_id] = []

            mapping[track_id].append(
                {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "team_id": team_id
                }
            )

            loaded_segments += 1

    # Sort each track's segments by start frame
    overlap_count = 0

    for track_id in mapping:
        mapping[track_id].sort(key=lambda x: x["start_frame"])

        # Count overlapping segments, if any
        segments = mapping[track_id]

        for i in range(len(segments) - 1):
            if segments[i]["end_frame"] >= segments[i + 1]["start_frame"]:
                overlap_count += 1

    print("Team mapping CSV loaded.")
    print(f"Loaded segments: {loaded_segments}")
    print(f"Skipped/unknown rows: {skipped_rows}")
    print(f"Mapped track IDs: {len(mapping)}")
    print(f"Overlapping segment warnings: {overlap_count}")

    return mapping


# ---------------------------------------------------------
# FIND TEAM FOR A SPECIFIC TRACK ID AND FRAME
# ---------------------------------------------------------

def find_team_for_frame(mapping, track_id, frame_id):
    """
    Find the best team label for a track_id at a specific frame_id.

    If multiple overlapping segments match, choose the shortest segment
    whose center is closest to the current frame.
    """
    segments = mapping.get(track_id)

    if not segments:
        return None

    best_team = None
    best_score = None

    for segment in segments:
        if segment["start_frame"] <= frame_id <= segment["end_frame"]:
            segment_length = segment["end_frame"] - segment["start_frame"]
            segment_center = (
                segment["start_frame"] + segment["end_frame"]
            ) / 2.0

            distance_to_center = abs(frame_id - segment_center)

            # Prefer shorter segments, then closer center
            score = (segment_length, distance_to_center)

            if best_score is None or score < best_score:
                best_score = score
                best_team = segment["team_id"]

    return best_team


# ---------------------------------------------------------
# APPLY TEAM MAPPING TO TRACKING JSON
# ---------------------------------------------------------

def apply_team_mapping(input_json_path, mapping, output_json_path):
    with open(input_json_path, "r") as f:
        payload = json.load(f)

    total_player_objects = 0
    assigned_player_objects = 0
    unknown_player_objects = 0

    max_blue_count = 0
    max_white_count = 0
    frames_with_too_many_players = 0

    for frame in payload.get("frames", []):
        frame_id = frame.get("frame_id")

        blue_count = 0
        white_count = 0

        for obj in frame.get("objects", []):
            if obj.get("class") != "player":
                continue

            total_player_objects += 1

            track_id = obj.get("track_id")

            assigned_team = find_team_for_frame(
                mapping,
                track_id,
                frame_id
            )

            obj["team_id"] = assigned_team

            if assigned_team == "team_blue":
                assigned_player_objects += 1
                blue_count += 1

            elif assigned_team == "team_white":
                assigned_player_objects += 1
                white_count += 1

            else:
                unknown_player_objects += 1

        max_blue_count = max(max_blue_count, blue_count)
        max_white_count = max(max_white_count, white_count)

        if (
            blue_count > MAX_PLAYERS_PER_TEAM or
            white_count > MAX_PLAYERS_PER_TEAM
        ):
            frames_with_too_many_players += 1

    # Add mapping metadata
    payload.setdefault("metadata", {})
    payload["metadata"]["team_mapping"] = {
        "teams": VALID_TEAMS,
        "mapped_track_ids": len(mapping),
        "total_player_objects": total_player_objects,
        "assigned_player_objects": assigned_player_objects,
        "unknown_player_objects": unknown_player_objects,
        "max_team_blue_players_in_one_frame": max_blue_count,
        "max_team_white_players_in_one_frame": max_white_count,
        "frames_with_more_than_allowed_team_players": frames_with_too_many_players,
        "max_players_per_team": MAX_PLAYERS_PER_TEAM
    }

    with open(output_json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("\nTeam mapping applied.")
    print(f"Total player objects: {total_player_objects}")
    print(f"Assigned player objects: {assigned_player_objects}")
    print(f"Unknown player objects: {unknown_player_objects}")
    print(f"Max team_blue players in one frame: {max_blue_count}")
    print(f"Max team_white players in one frame: {max_white_count}")
    print(f"Frames with more than {MAX_PLAYERS_PER_TEAM} players in one team: {frames_with_too_many_players}")
    print(f"\nSaved: {output_json_path}")


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

team_mapping = load_team_mapping(TEAM_CSV)

apply_team_mapping(
    INPUT_JSON,
    team_mapping,
    OUTPUT_JSON
)