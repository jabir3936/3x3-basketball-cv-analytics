import json
import os
import csv
import cv2


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

INPUT_JSON = "data/tracking/archive/match_tracking_data_interpolated_v7.json"
VIDEO_PATH = "media/footage/court_footage.mp4"

CROP_DIR = "player_crops"
CSV_PATH = "data/team_mapping/team_mapping_template.csv"

# Number of jersey crop images to save per player segment
SAMPLES_PER_SEGMENT = 3

# If a track disappears for more than this many frames,
# start a new segment.
MAX_SEGMENT_GAP_FRAMES = 10

# Ignore very short track segments.
# Increase to 10 if you get too many tiny segments.
MIN_SEGMENT_FRAMES = 5


# ---------------------------------------------------------
# BUILD PLAYER TRACK SEGMENTS
# ---------------------------------------------------------

def build_player_segments(payload):
    """
    Build player segments from tracking JSON.

    A segment is a continuous appearance of one track_id.
    Segments are split when:
        - there is an invalid court view / cut scene
        - the track disappears for too long
    """
    segments = []
    active = {}
    scene_block = 0

    def close_segment(track_id):
        if track_id in active:
            seg = active.pop(track_id)

            if seg["frame_count"] >= MIN_SEGMENT_FRAMES:
                segments.append(seg)

    def close_all_segments():
        for track_id in list(active.keys()):
            close_segment(track_id)

    for frame in payload.get("frames", []):
        frame_id = frame.get("frame_id")
        is_valid = frame.get("is_valid_court_view", True)

        # Split segments on cut scenes / invalid frames
        if not is_valid:
            close_all_segments()
            scene_block += 1
            continue

        for obj in frame.get("objects", []):
            if obj.get("class") != "player":
                continue

            track_id = obj.get("track_id")
            bbox = obj.get("bbox")

            if track_id is None or bbox is None:
                continue

            # If track exists but disappeared for too long, close it
            if track_id in active:
                gap = frame_id - active[track_id]["last_frame"]

                if gap > MAX_SEGMENT_GAP_FRAMES:
                    close_segment(track_id)

            # Start new segment
            if track_id not in active:
                active[track_id] = {
                    "track_id": track_id,
                    "scene_block": scene_block,
                    "start_frame": frame_id,
                    "end_frame": frame_id,
                    "last_frame": frame_id,
                    "frame_count": 1,
                    "samples": [
                        (frame_id, bbox)
                    ]
                }

            # Update existing segment
            else:
                active[track_id]["end_frame"] = frame_id
                active[track_id]["last_frame"] = frame_id
                active[track_id]["frame_count"] += 1
                active[track_id]["samples"].append((frame_id, bbox))

    close_all_segments()

    # Assign segment IDs and choose sample frames
    final_segments = []

    for seg in segments:
        segment_id = (
            f"{seg['track_id']}_"
            f"{seg['scene_block']}_"
            f"{seg['start_frame']}_"
            f"{seg['end_frame']}"
        )

        samples = seg["samples"]
        n = len(samples)

        if n <= SAMPLES_PER_SEGMENT:
            chosen_samples = samples
        else:
            indexes = [
                int(i * (n - 1) / (SAMPLES_PER_SEGMENT - 1))
                for i in range(SAMPLES_PER_SEGMENT)
            ]

            chosen_samples = [samples[i] for i in indexes]

        seg["segment_id"] = segment_id
        seg["chosen_samples"] = chosen_samples

        final_segments.append(seg)

    return final_segments


# ---------------------------------------------------------
# CROP JERSEY REGION
# ---------------------------------------------------------

def get_jersey_crop(frame, bbox):
    """
    Crop the upper/middle body region where the jersey is usually visible.
    """
    x1, y1, x2, y2 = map(int, bbox)

    h = y2 - y1
    w = x2 - x1

    if h <= 0 or w <= 0:
        return None

    # Jersey region:
    # top 20% to 75% of bbox height
    # left 15% to right 85% of bbox width
    top = int(y1 + 0.20 * h)
    bottom = int(y1 + 0.75 * h)
    left = int(x1 + 0.15 * w)
    right = int(x2 - 0.15 * w)

    frame_h, frame_w = frame.shape[:2]

    top = max(0, top)
    bottom = min(frame_h, bottom)
    left = max(0, left)
    right = min(frame_w, right)

    crop = frame[top:bottom, left:right]

    # Fallback to full bbox if crop is invalid
    if crop.size == 0:
        crop = frame[y1:y2, x1:x2]

    return crop


# ---------------------------------------------------------
# EXPORT CROPS
# ---------------------------------------------------------

def export_player_crops(video_path, segments):
    os.makedirs(CROP_DIR, exist_ok=True)

    needed_frames = {}

    for seg in segments:
        for frame_id, bbox in seg["chosen_samples"]:
            if frame_id not in needed_frames:
                needed_frames[frame_id] = []

            needed_frames[frame_id].append(
                {
                    "segment_id": seg["segment_id"],
                    "track_id": seg["track_id"],
                    "bbox": bbox
                }
            )

    if not needed_frames:
        print("No player crops to export.")
        return

    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    max_needed_frame = max(needed_frames.keys())

    while frame_id <= max_needed_frame:
        ret, frame = cap.read()

        if not ret:
            break

        frame_id += 1

        if frame_id not in needed_frames:
            continue

        for item in needed_frames[frame_id]:
            crop = get_jersey_crop(frame, item["bbox"])

            if crop is None or crop.size == 0:
                continue

            filename = f"{item['segment_id']}_frame_{frame_id}.jpg"
            save_path = os.path.join(CROP_DIR, filename)

            cv2.imwrite(save_path, crop)

    cap.release()

    print(f"Player crops saved to: {CROP_DIR}")


# ---------------------------------------------------------
# CREATE TEAM MAPPING CSV TEMPLATE
# ---------------------------------------------------------

def write_team_mapping_template(segments):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "segment_id",
            "track_id",
            "scene_block",
            "start_frame",
            "end_frame",
            "frame_count",
            "sample_images",
            "team_id"
        ])

        for seg in segments:
            sample_files = [
                f"{seg['segment_id']}_frame_{frame_id}.jpg"
                for frame_id, _ in seg["chosen_samples"]
            ]

            writer.writerow([
                seg["segment_id"],
                seg["track_id"],
                seg["scene_block"],
                seg["start_frame"],
                seg["end_frame"],
                seg["frame_count"],
                ";".join(sample_files),
                ""
            ])

    print(f"Team mapping template saved to: {CSV_PATH}")


# ---------------------------------------------------------
# RUN TEAM MAPPING STEP 1
# ---------------------------------------------------------

with open(INPUT_JSON, "r") as f:
    payload = json.load(f)

segments = build_player_segments(payload)

print(f"Built {len(segments)} player segments.")

export_player_crops(VIDEO_PATH, segments)
write_team_mapping_template(segments)