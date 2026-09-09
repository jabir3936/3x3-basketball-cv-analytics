import json
from collections import Counter

with open("data/tracking/latest/match_tracking_data_official.json") as f:
    frames = json.load(f)["frames"]

teams = Counter()
combos = Counter()
for fr in frames:
    if not fr.get("is_valid_court_view", True):
        continue
    b = w = g = 0
    for obj in fr.get("objects", []):
        if obj["class"] != "player":
            continue
        t = obj.get("team_id")
        if t == "team_blue": b += 1; teams["blue"] += 1
        elif t == "team_white": w += 1; teams["white"] += 1
        else: g += 1; teams["gray"] += 1
    combos[(b, w, g)] += 1

total = sum(teams.values())
good = sum(c for (b, w, g), c in combos.items() if b >= 3 and w >= 3)
n_frames = sum(combos.values())

print(f"blue {100*teams['blue']/total:.1f}% | white {100*teams['white']/total:.1f}% | gray {100*teams['gray']/total:.1f}%")
print("Top combos:", combos.most_common(3))
print(f"Frames with 3+ blue AND 3+ white: {100*good/n_frames:.1f}%")