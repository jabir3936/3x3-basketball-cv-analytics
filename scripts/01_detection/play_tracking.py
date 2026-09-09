import cv2
import json
import os

def play_local_tracking(video_path, json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing tracking blueprint data at: {json_path}")

    # Load the tracking data compiled by Colab
    print("Loading tracking telemetry coordinates...")
    with open(json_path, 'r') as f:
        tracking_data = json.load(f)
        
    # Map frame IDs to their object lists for instant O(1) lookups during playback
    frame_map = {f["frame_id"]: f["objects"] for f in tracking_data["frames"]}

    cap = cv2.VideoCapture(video_path)
    frame_id = 1

    print("Igniting high-speed local rendering loop. Press 'q' to exit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check if we have tracking data for this frame
        if frame_id in frame_map:
            for obj in frame_map[frame_id]:
                x1, y1, x2, y2 = obj["bbox"]
                track_id = obj["track_id"]
                label = obj["class"]
                
                # Assign distinct UI colors: Neon Green for Players, Bright Yellow for the Ball
                color = (0, 255, 0) if label == "player" else (0, 255, 255)
                
                # 1. Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 2. Draw the tracking ID text above the asset
                text = f"{label.upper()} #{track_id}"
                cv2.putText(frame, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 3. Optional: Draw a dot at the calculated anchor coordinates
                if "feet_coord_px" in obj:
                    cv2.circle(frame, tuple(obj["feet_coord_px"]), 5, (0, 0, 255), -1)
                elif "center_coord_px" in obj:
                    cv2.circle(frame, tuple(obj["center_coord_px"]), 5, (0, 0, 255), -1)

        # Draw the global frame counter onto the interface
        cv2.putText(frame, f"Frame: {frame_id}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Render the frame to your display
        cv2.imshow("Production Sport Analytics Engine - Local Playback", frame)
        
        # Controls playback speed (~30ms delay roughly matches 30 FPS video)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Playback loop closed cleanly.")

# Run the local player
play_local_tracking("media/footage/calibration_test.mov", "match_tracking_data.json")