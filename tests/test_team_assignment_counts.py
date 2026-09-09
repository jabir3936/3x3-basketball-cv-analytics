import json
from collections import Counter

with open("data/tracking/latest/match_tracking_data_official.json") as f:
    frames = json.load(f)["frames"]

teams = Counter()
per_frame_combos = Counter()

for fr in frames:
    if not fr.get("is_valid_court_view", True):
        continue
    blue = white = gray = 0
    for obj in fr.get("objects", []):
        if obj["class"] != "player":
            continue
        t = obj.get("team_id")
        if t == "team_blue":
            blue += 1; teams["team_blue"] += 1
        elif t == "team_white":
            white += 1; teams["team_white"] += 1
        else:
            gray += 1; teams["unlabeled"] += 1
    per_frame_combos[(blue, white, gray)] += 1

total = sum(teams.values())
print("=== TEST 3: TEAM LABEL COVERAGE ===")
print(f"  team_blue:  {teams['team_blue']}  ({100*teams['team_blue']/total:.1f}%)")
print(f"  team_white: {teams['team_white']}  ({100*teams['team_white']/total:.1f}%)")
print(f"  unlabeled:  {teams['unlabeled']}  ({100*teams['unlabeled']/total:.1f}%)")
print(f"\n  Top 5 (blue, white, gray) combos per frame:")
for combo, count in per_frame_combos.most_common(5):
    print(f"    {combo}: {count} frames")
ideal = sum(1 for (b,w,g), c in per_frame_combos.items() if b >= 3 and w >= 3 for _ in range(c))
total_frames = sum(per_frame_combos.values())
print(f"\n  Frames with >=3 blue AND >=3 white: {ideal}/{total_frames} ({100*ideal/total_frames:.1f}%)")
print(f"  [{'PASS' if ideal/total_frames > 0.50 else 'FAIL'}]")