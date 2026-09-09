import json, csv

TEAMS_JSON = "data/tracking/latest/match_tracking_data_teams_new_7Sep.json"   # Script 2's output
AUTO_CSV = "data/team_mapping/team_mapping_auto.csv"

segments = []
with open(AUTO_CSV) as f:
    for row in csv.DictReader(f):
        if row["team_id"] in ("team_blue", "team_white"):
            segments.append({"track_id": int(row["track_id"]),
                             "start": int(row["start_frame"]),
                             "end": int(row["end_frame"]),
                             "team": row["team_id"]})

with open(TEAMS_JSON) as f:
    payload = json.load(f)
frames = payload["frames"]

def closeness(b, w):
    return -(abs(b - 3) + abs(w - 3))   # 0 when exactly 3 vs 3

for it in range(2):
    counts, members = {}, {}
    for fr in frames:
        if not fr.get("is_valid_court_view", True):
            continue
        b = w = 0
        mem = []
        for obj in fr.get("objects", []):
            if obj.get("class") != "player":
                continue
            t = obj.get("team_id")
            if t == "team_blue": b += 1
            elif t == "team_white": w += 1
            mem.append((obj, t))
        counts[fr["frame_id"]] = [b, w]
        members[fr["frame_id"]] = mem

    flips = []
    for s in segments:
        cur = flp = n = 0
        for fid in range(s["start"], s["end"] + 1):
            if fid not in counts:
                continue
            if not any(o.get("track_id") == s["track_id"] and t == s["team"]
                       for o, t in members[fid]):
                continue
            b, w = counts[fid]
            n += 1
            cur += closeness(b, w)
            flp += closeness(b - 1, w + 1) if s["team"] == "team_blue" \
                   else closeness(b + 1, w - 1)
        if n >= 15 and flp - cur > 0:
            flips.append((flp - cur, s))

    flips.sort(key=lambda x: -x[0])
    for delta, s in flips:
        new_team = "team_white" if s["team"] == "team_blue" else "team_blue"
        for fr in frames:
            fid = fr["frame_id"]
            if not (s["start"] <= fid <= s["end"]):
                continue
            for obj in fr.get("objects", []):
                if (obj.get("class") == "player"
                        and obj.get("track_id") == s["track_id"]
                        and obj.get("team_id") == s["team"]):
                    obj["team_id"] = new_team
        s["team"] = new_team
    print(f"Pass {it}: flipped {len(flips)} bad segments")

with open(TEAMS_JSON, "w") as f:
    json.dump(payload, f, indent=2)
print("Saved repaired", TEAMS_JSON)