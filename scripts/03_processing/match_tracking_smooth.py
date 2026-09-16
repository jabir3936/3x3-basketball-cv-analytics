"""
Smooth player and ball court coordinates using a bidirectional
Savitzky-Golay filter (zero phase delay, offline-only).

Replaces the previous forward-only EMA which introduced ~62 cm of
positional lag at sprint speed.

Usage (from project root):
    python scripts/03_processing/match_tracking_smooth.py
"""
import json
import numpy as np
from scipy.signal import savgol_filter

INPUT_JSON = "data/tracking/latest/match_tracking_data_official.json"
OUTPUT_JSON = "data/tracking/latest/match_tracking_data_smooth.json"

# Savitzky-Golay parameters
PLAYER_WINDOW = 11   # must be odd; ~0.37s at 30 fps
PLAYER_POLYORDER = 3
BALL_WINDOW = 7      # tighter window for ball (fewer valid samples)
BALL_POLYORDER = 2
MAX_GAP = 5          # max frame gap before splitting into new segment


def smooth_segments(segments, window, polyorder):
    """Apply Savitzky-Golay to each contiguous segment of (frame_id, x, y)."""
    smoothed = {}  # frame_id -> [x, y]
    for seg in segments:
        if len(seg) < 3:
            # Too short to smooth — keep raw
            for fid, x, y in seg:
                smoothed[fid] = [round(x, 3), round(y, 3)]
            continue

        fids = [s[0] for s in seg]
        xs = np.array([s[1] for s in seg])
        ys = np.array([s[2] for s in seg])

        # Window must be odd and <= segment length
        win = min(window, len(seg))
        if win % 2 == 0:
            win -= 1
        # Polyorder must be < window
        poly = min(polyorder, win - 1)

        if win >= 3 and poly >= 1:
            xs_smooth = savgol_filter(xs, win, poly, mode="nearest")
            ys_smooth = savgol_filter(ys, win, poly, mode="nearest")
        else:
            xs_smooth = xs
            ys_smooth = ys

        for i, fid in enumerate(fids):
            smoothed[fid] = [round(float(xs_smooth[i]), 3),
                             round(float(ys_smooth[i]), 3)]
    return smoothed


def build_segments(entries, max_gap):
    """Split a list of (frame_id, x, y) into contiguous segments."""
    if not entries:
        return []
    entries.sort(key=lambda e: e[0])
    segments = []
    current = [entries[0]]
    for e in entries[1:]:
        if e[0] - current[-1][0] <= max_gap:
            current.append(e)
        else:
            segments.append(current)
            current = [e]
    segments.append(current)
    return segments


def main():
    with open(INPUT_JSON) as f:
        payload = json.load(f)

    # Collect per-track raw coordinates
    track_data = {}   # track_id -> [(frame_id, x, y), ...]
    ball_data = []     # [(frame_id, x, y), ...]

    # Track which frames are invalid (circuit breaker) so we split segments
    invalid_frames = set()

    for fr in payload["frames"]:
        fid = fr["frame_id"]
        if not fr.get("is_valid_court_view", True):
            invalid_frames.add(fid)
            continue
        for obj in fr.get("objects", []):
            court = obj.get("court_coords_meters")
            if not court or obj.get("court_coords_valid") is False:
                continue
            if obj["class"] == "player":
                tid = obj["track_id"]
                track_data.setdefault(tid, []).append((fid, court[0], court[1]))
            elif obj["class"] == "ball":
                ball_data.append((fid, court[0], court[1]))

    # Build segments and smooth
    player_smooth = {}  # track_id -> {frame_id -> [x, y]}
    for tid, entries in track_data.items():
        segs = build_segments(entries, MAX_GAP)
        # Further split at invalid frame boundaries
        split_segs = []
        for seg in segs:
            current = []
            for e in seg:
                # Check if any frame between the last entry and this one was invalid
                if current and any(f in invalid_frames
                                   for f in range(current[-1][0] + 1, e[0])):
                    split_segs.append(current)
                    current = [e]
                else:
                    current.append(e)
            if current:
                split_segs.append(current)
        player_smooth[tid] = smooth_segments(split_segs, PLAYER_WINDOW, PLAYER_POLYORDER)

    ball_segs = build_segments(ball_data, MAX_GAP)
    ball_smooth = smooth_segments(ball_segs, BALL_WINDOW, BALL_POLYORDER)

    # Write smoothed coordinates back
    n_smoothed = 0
    for fr in payload["frames"]:
        fid = fr["frame_id"]
        if not fr.get("is_valid_court_view", True):
            continue
        for obj in fr.get("objects", []):
            if obj.get("court_coords_valid") is False:
                continue
            court = obj.get("court_coords_meters")
            if not court:
                continue
            if obj["class"] == "player":
                tid = obj["track_id"]
                if tid in player_smooth and fid in player_smooth[tid]:
                    obj["court_coords_meters"] = player_smooth[tid][fid]
                    n_smoothed += 1
            elif obj["class"] == "ball":
                if fid in ball_smooth:
                    obj["court_coords_meters"] = ball_smooth[fid]
                    n_smoothed += 1

    with open(OUTPUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {OUTPUT_JSON}")
    print(f"Smoothed {n_smoothed} coordinate entries "
          f"(players: window={PLAYER_WINDOW}, poly={PLAYER_POLYORDER}; "
          f"ball: window={BALL_WINDOW}, poly={BALL_POLYORDER})")


if __name__ == "__main__":
    main()