import json

INPUT_JSON = "data/tracking/latest/match_tracking_data_official.json"
OUTPUT_JSON = "data/tracking/latest/match_tracking_data_smooth.json"

PLAYER_ALPHA = 0.35   # lower = smoother
BALL_ALPHA = 0.45
MAX_GAP = 5

with open(INPUT_JSON) as f:
    payload = json.load(f)

last = {}
for fr in payload["frames"]:
    fid = fr["frame_id"]
    if not fr.get("is_valid_court_view", True):
        last = {}
        continue
    seen = set()
    for obj in fr.get("objects", []):
        cls = obj["class"]
        key = obj["track_id"] if cls == "player" else "ball"
        court = obj.get("court_coords_meters")
        if not court or obj.get("court_coords_valid") is False:
            continue
        seen.add(key)
        prev = last.get(key)
        if prev and fid - prev[0] <= MAX_GAP:
            a = BALL_ALPHA if cls == "ball" else PLAYER_ALPHA
            court = [round(a*court[0] + (1-a)*prev[1][0], 3),
                     round(a*court[1] + (1-a)*prev[1][1], 3)]
            obj["court_coords_meters"] = court
        last[key] = (fid, court)
    for key in list(last):
        if key not in seen and fid - last[key][0] > MAX_GAP:
            del last[key]

with open(OUTPUT_JSON, "w") as f:
    json.dump(payload, f, indent=2)
print("Saved", OUTPUT_JSON)