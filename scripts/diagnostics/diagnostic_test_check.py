import json, math
from collections import Counter, defaultdict

TRACKING_JSON = "data/tracking/latest/match_tracking_data_official.json"
FSM_JSON = "data/fsm/match_fsm_data.json"

with open(TRACKING_JSON) as f:
    track = json.load(f)
with open(FSM_JSON) as f:
    fsm = json.load(f)

fsm_by_id = {fr["frame_id"]: fr.get("fsm", {}) for fr in fsm["frames"]}

player_count_hist = Counter()
team_combos = Counter()
skipped_invalid = 0
off_canvas = 0
frames_missing_teams = 0
per_track = defaultdict(Counter)
ball_poss_dists = []

for fr in track["frames"]:
    if not fr.get("is_valid_court_view", True):
        continue
    fid = fr["frame_id"]
    players = [o for o in fr.get("objects", []) if o["class"] == "player"]
    player_count_hist[len(players)] += 1

    blue = white = unknown = 0
    for o in players:
        team = o.get("team_id")
        court = o.get("court_coords_meters")
        valid = o.get("court_coords_valid", True)

        if team == "team_blue": blue += 1
        elif team == "team_white": white += 1
        else: unknown += 1

        per_track[o["track_id"]][team or "UNLABELED"] += 1

        if valid is False:
            skipped_invalid += 1
        elif court and (court[0] < -8 or court[0] > 9 or court[1] < -1 or court[1] > 16):
            off_canvas += 1

    team_combos[(blue, white, unknown)] += 1
    if blue < 3 or white < 3:
        frames_missing_teams += 1

    # Ball-vs-possessor offset (quantifies the "ball slightly off" issue)
    info = fsm_by_id.get(fid, {})
    poss = info.get("possession_player_id")
    if poss is not None:
        ball = next((o for o in fr["objects"] if o["class"] == "ball"), None)
        p = next((o for o in players if o["track_id"] == poss), None)
        if ball and p and ball.get("court_coords_meters") and p.get("court_coords_meters"):
            ball_poss_dists.append(
                math.dist(ball["court_coords_meters"], p["court_coords_meters"])
            )

print("=== DIAGNOSTIC REPORT ===")
print("Players-per-frame histogram:", dict(sorted(player_count_hist.items())))
print("Frames with <3 blue OR <3 white on court:", frames_missing_teams)
print("Most common (blue, white, unknown) combos:", team_combos.most_common(6))
print("Objects skipped (court_coords_valid=False):", skipped_invalid)
print("Objects outside drawable canvas:", off_canvas)
if ball_poss_dists:
    print(f"Ball-possessor offset: mean {sum(ball_poss_dists)/len(ball_poss_dists):.2f} m, "
          f"max {max(ball_poss_dists):.2f} m")

print("\nTrack-id team usage (shows unlabeled/gray players):")
for tid, c in sorted(per_track.items(), key=lambda kv: -sum(kv[1].values()))[:25]:
    print(f"  ID {tid}: {dict(c)}")