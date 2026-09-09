import cv2, json

with open("data/tracking/latest/match_tracking_data_official.json") as f:
    fr = {f["frame_id"]: f for f in json.load(f)["frames"]}[1000]

cap = cv2.VideoCapture("media/footage/court_footage.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 999)
ret, frame = cap.read()
cap.release()

# Draw bboxes from JSON onto the video frame
for obj in fr.get("objects", []):
    x1,y1,x2,y2 = map(int, obj["bbox"])
    color = (0,255,0) if obj["class"]=="player" else (0,165,255)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    cv2.putText(frame, f"{obj['track_id']}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.imwrite("media/checks/test5_sync_check.jpg", frame)
print("=== TEST 5: VIDEO SYNC ===")
print("Saved media/checks/test5_sync_check.jpg")
print("Check: do the green boxes sit exactly on the players?")
print("[PASS] if boxes match players, [FAIL] if boxes are on wrong people or shifted")