import cv2, json, math
import numpy as np

TRACKING_JSON = "data/tracking/latest/match_tracking_data_official.json"
VIDEO_PATH = "media/footage/court_footage.mp4"
TEST_FRAME_ID = 1000  # Pick a frame where all players are visible

with open(TRACKING_JSON) as f:
    all_frames = {fr["frame_id"]: fr for fr in json.load(f)["frames"]}

fr = all_frames[TEST_FRAME_ID]

# Read the actual video frame
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, TEST_FRAME_ID - 1)
ret, video_frame = cap.read()
cap.release()

# Draw 2D map for this one frame
SCALE = 40.0; MIN_X = -8.0; MIN_Y = -1.0; MARGIN = 40
CANVAS_W = 720; CANVAS_H = 720
TEAM_COLORS = {"team_blue": (255,130,0), "team_white": (0,0,230), None: (160,160,160)}

def c2c(x, y):
    return (int(MARGIN+(x-MIN_X)*SCALE), int(MARGIN+(y-MIN_Y)*SCALE))

canvas = 35*np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

# Court lines
for a,b in [((-7.5,0),(7.5,0)),((7.5,0),(7.5,11)),((7.5,11),(-7.5,11)),((-7.5,11),(-7.5,0))]:
    cv2.line(canvas, c2c(*a), c2c(*b), (80,80,80), 2)
for a,b in [((-2.45,0),(2.45,0)),((2.45,0),(2.45,5.8)),((2.45,5.8),(-2.45,5.8)),((-2.45,5.8),(-2.45,0))]:
    cv2.line(canvas, c2c(*a), c2c(*b), (80,80,80), 2)
prev = None
for deg in range(0,181,3):
    t = math.radians(deg)
    p = c2c(6.75*math.cos(t), 1.575+6.75*math.sin(t))
    if prev: cv2.line(canvas, prev, p, (100,100,255), 2)
    prev = p
cv2.circle(canvas, c2c(0, 1.575), 6, (0,0,255), -1)

# Players
print(f"=== TEST 4: FRAME {TEST_FRAME_ID} PLAYER POSITIONS ===")
for obj in fr.get("objects", []):
    if obj["class"] != "player": continue
    court = obj.get("court_coords_meters")
    if not court: continue
    tid = obj["track_id"]
    team = obj.get("team_id")
    color = TEAM_COLORS.get(team, (160,160,160))
    px = c2c(court[0], court[1])
    cv2.circle(canvas, px, 14, color, -1)
    cv2.putText(canvas, str(tid), (px[0]-12,px[1]+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    print(f"  ID {tid:5d}  team={str(team):12s}  court=({court[0]:6.2f}, {court[1]:6.2f})  bbox={obj['bbox']}")

# Ball
ball = next((o for o in fr["objects"] if o["class"]=="ball"), None)
if ball and ball.get("court_coords_meters"):
    bc = ball["court_coords_meters"]
    cv2.circle(canvas, c2c(bc[0], bc[1]), 7, (0,220,0), -1)
    print(f"  BALL     court=({bc[0]:6.2f}, {bc[1]:6.2f})  bbox={ball['bbox']}")

# Save both
cv2.imwrite("media/checks/test4_video_frame.jpg", video_frame)
cv2.imwrite("media/checks/test4_2d_map.jpg", canvas)
print("\nSaved media/checks/test4_video_frame.jpg and media/checks/test4_2d_map.jpg")
print("Compare them side by side: do the dots match the player positions?")