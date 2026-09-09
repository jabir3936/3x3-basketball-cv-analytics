import cv2, json, csv

TEAMS_JSON = "data/tracking/latest/match_tracking_data_teams.json"
VIDEO_PATH = "media/footage/court_footage.mp4"
OUT_CSV = "data/team_mapping/team_mapping_auto.csv"

MIN_SEGMENT_FRAMES = 15
SAMPLES_PER_SEGMENT = 5

with open(TEAMS_JSON) as f:
    frames = json.load(f)["frames"]

# ---- 1) collect unlabeled runs per track_id ----
runs, segments = {}, []
def close_run(tid):
    if tid in runs and runs[tid]["count"] >= MIN_SEGMENT_FRAMES:
        segments.append(runs[tid])
    runs.pop(tid, None)

for fr in frames:
    fid = fr["frame_id"]
    if not fr.get("is_valid_court_view", True):
        for t in list(runs): close_run(t)
        continue
    present = set()
    for obj in fr.get("objects", []):
        if obj["class"] != "player": continue
        tid = obj["track_id"]; present.add(tid)
        if obj.get("team_id") is not None:
            close_run(tid); continue
        if tid in runs and fid - runs[tid]["last"] > 10: close_run(tid)
        if tid not in runs:
            runs[tid] = {"track_id": tid, "start": fid, "last": fid, "count": 0, "samples": []}
        r = runs[tid]; r["last"] = fid; r["count"] += 1
        r["samples"].append((fid, obj["bbox"]))
    for tid in list(runs):
        if tid not in present and fid - runs[tid]["last"] > 10: close_run(tid)
for t in list(runs): close_run(t)

needed = {}
for seg in segments:
    s = seg["samples"]; n = len(s)
    idx = [int(i*(n-1)/(SAMPLES_PER_SEGMENT-1)) for i in range(SAMPLES_PER_SEGMENT)] if n > 1 else [0]
    seg["chosen"] = [s[i] for i in idx]
    for fid, bbox in seg["chosen"]:
        needed.setdefault(fid, []).append((seg, bbox))

# ---- 2) read video, measure jersey brightness ----
cap = cv2.VideoCapture(VIDEO_PATH)
fid, max_fid = 0, max(needed) if needed else 0
while fid <= max_fid:
    ret, frame = cap.read()
    if not ret: break
    fid += 1
    if fid not in needed: continue
    for seg, bbox in needed[fid]:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = y2-y1, x2-x1
        crop = frame[int(y1+0.25*h):int(y1+0.70*h), int(x1+0.30*w):int(x1+0.70*w)]
        if crop.size == 0: continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        seg.setdefault("S", []).append(float(hsv[...,1].mean()))
        seg.setdefault("V", []).append(float(hsv[...,2].mean()))
cap.release()

# ---- 3) classify and write CSV ----
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["segment_id","track_id","start_frame","end_frame","team_id","mean_S","mean_V"])
    n_written = 0
    for seg in segments:
        if "V" not in seg: continue
        V = sum(seg["V"])/len(seg["V"]); S = sum(seg["S"])/len(seg["S"])
        if V >= 150:   team = "team_white"   # bright jersey
        elif V >= 60:  team = "team_blue"    # dark blue jersey
        else:          team = "unknown"      # black referee etc.
        w.writerow([f"{seg['track_id']}_{seg['start']}_{seg['last']}",
                    seg["track_id"], seg["start"], seg["last"], team, round(S,1), round(V,1)])
        n_written += 1
print(f"Wrote {OUT_CSV} with {n_written} auto-labeled segments")