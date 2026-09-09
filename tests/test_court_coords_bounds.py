import json

with open("data/tracking/latest/match_tracking_data_official.json") as f:
    frames = json.load(f)["frames"]

total = 0
inside = 0
outside = 0
invalid = 0

for fr in frames:
    if not fr.get("is_valid_court_view", True):
        continue
    for obj in fr.get("objects", []):
        if obj["class"] != "player":
            continue
        total += 1
        court = obj.get("court_coords_meters")
        valid = obj.get("court_coords_valid", True)
        if not valid or court is None:
            invalid += 1
            continue
        x, y = court
        if -8.0 <= x <= 8.0 and -1.0 <= y <= 12.0:
            inside += 1
        else:
            outside += 1

print("=== TEST 2: RE-PROJECTION BOUNDS ===")
print(f"  Total player objects: {total}")
print(f"  Inside court bounds:  {inside}  ({100*inside/total:.1f}%)")
print(f"  Outside court bounds: {outside}  ({100*outside/total:.1f}%)")
print(f"  Invalid coordinates:  {invalid}")
print(f"  [{'PASS' if inside/total > 0.90 else 'FAIL'}]")