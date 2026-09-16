import cv2
import json
from pathlib import Path

ROOT = Path('.').resolve()
TRACKING_JSON = ROOT / "data/tracking/latest/match_tracking_data_smooth.json"

with TRACKING_JSON.open("r") as f:
    tracking_frames = {fr["frame_id"]: fr for fr in json.load(f)["frames"]}

track_team_vote = {}
for fr in tracking_frames.values():
    for obj in fr.get("objects", []):
        if obj.get("class") == "player" and obj.get("team_id"):
            tid = obj.get("track_id")
            track_team_vote.setdefault(tid, {}).setdefault(obj["team_id"], 0)
            track_team_vote[tid][obj["team_id"]] += 1

dominant_team_map = {
    tid: max(votes, key=votes.get)
    for tid, votes in track_team_vote.items()
}

TEAM_COLORS = {
    "team_blue": (255, 130, 0),
    "team_white": (0, 0, 230),
    "unknown": (128, 128, 128),
}

# simulate what render_with_2dmap does on frame 961
track_fr = tracking_frames.get(961, {})
print("Before select_display_players:")
for obj in track_fr.get("objects", []):
    if obj["class"] == "player":
        tid = obj["track_id"]
        team = dominant_team_map.get(tid, obj.get("team_id"))
        color = TEAM_COLORS.get(team, (128, 128, 128))
        print(f"  TID {tid}: dominant_team={team}, color={color}")

