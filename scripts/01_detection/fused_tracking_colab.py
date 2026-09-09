#RT-DETR implementation with tracknet (No transfer learning)
import os
import json
import cv2
import numpy as np
import pandas as pd
import subprocess
from ultralytics import RTDETR

# ---------------------------------------------------------
# HOMOGRAPHY GEOMETRIC COORDINATE TRANSFORMATION
# ---------------------------------------------------------
def transform_pixel_to_meters(H, pixel_point):
    x, y = pixel_point
    vec = np.array([x, y, 1.0], dtype=np.float32)
    transformed = np.dot(H, vec)
    w = transformed[2]
    if w == 0: return [0.0, 0.0]
    return [float(transformed[0] / w), float(transformed[1] / w)]

# ---------------------------------------------------------
# FUSED EXECUTION PIPELINE (TrackNetV3 Native -> RT-DETR)
# ---------------------------------------------------------
def run_fused_pipeline(video_path, calibration_path, output_json_path, tracknet_ckpt, inpaintnet_ckpt):

    # =========================================================
    # STEP 1: NATIVE TRACKNETV3 EXECUTION
    # =========================================================
    print(f"🎬 Step 1: Running TrackNetV3 Native Prediction on {video_path}...")

    pred_dir = "temp_prediction"
    os.makedirs(pred_dir, exist_ok=True)

    # Execute the repository's native predict script via subprocess
    cmd = [
        'python', 'predict.py',
        '--video_file', video_path,
        '--tracknet_file', tracknet_ckpt,
        '--inpaintnet_file', inpaintnet_ckpt,
        '--save_dir', pred_dir
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f" TrackNetV3 execution failed. Please check paths and dependencies. {e}")
        return

    # TrackNetV3 saves output as a CSV based on the video name
    csv_files = [f for f in os.listdir(pred_dir) if f.endswith('.csv')]
    if not csv_files:
        print(" Could not find TrackNetV3 output CSV. Prediction may have failed silently.")
        return

    tracknet_csv_path = os.path.join(pred_dir, csv_files[0])
    print(f" TrackNetV3 finished. Ball trajectory saved to {tracknet_csv_path}")

    # Load ball coordinates into memory
    print(" Loading ball coordinates into memory...")
    ball_df = pd.read_csv(tracknet_csv_path)

    ball_trajectory = {}
    for index, row in ball_df.iterrows():
        frame_idx = int(row['Frame'])
        vis = int(row.get('Visibility', 1))

        # If Visibility is > 0, the ball was detected or inpainted by TrackNet
        if vis > 0:
            ball_trajectory[frame_idx] = (int(row['X']), int(row['Y']))


    # =========================================================
    # STEP 2: RT-DETR PLAYER ANALYTICS & HOMOGRAPHY FUSION
    # =========================================================
    print("🎬 Step 2: Booting RT-DETR Player Analytics & Fusing Telemetry...")

    with open(calibration_path, 'r') as f:
        calib_data = json.load(f)
    H = np.array(calib_data["homography_matrix"], dtype=np.float32)

    player_model = RTDETR("models/rtdetr-l.pt")

    cap = cv2.VideoCapture(video_path)
    ORIG_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ORIG_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    CLOSE_UP_THRESHOLD = ORIG_H * 0.35
    MIN_PLAYERS_REQUIRED = 4

    tracking_payload = {
        "metadata": {
            "video_name": os.path.basename(video_path),
            "frame_width": ORIG_W,
            "frame_height": ORIG_H,
            "total_frames": 0
        },
        "frames": []
    }

    # OpenCV is 0-indexed, matching TrackNet's standard CSV output
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        raw_frame_objects = []

        # --- 1. BALL HOMOGRAPHY BOUNCER ---
        if frame_idx in ball_trajectory:
            bx, by = ball_trajectory[frame_idx]
            bx_m, by_m = transform_pixel_to_meters(H, (bx, by))

            # The ball can briefly leave the court lines.
            if -12.0 <= bx_m <= 12.0 and -3.0 <= by_m <= 15.0:
                raw_frame_objects.append({
                    "track_id": 999,
                    "class": "ball",
                    "center_coord_px": [bx, by],
                    "court_coords_meters": [round(bx_m, 3), round(by_m, 3)]
                })

        # --- 2. PLAYER HOMOGRAPHY BOUNCER ---
        results = player_model.track(source=frame, tracker="botsort.yaml", classes=[0], persist=True, verbose=False)
        r = results[0]

        player_count = 0
        is_close_up = False

        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            confidences = r.boxes.conf.cpu().numpy()

            for box, track_id, conf in zip(boxes, ids, confidences):
                x1, y1, x2, y2 = map(int, box)

                if (y2 - y1) > CLOSE_UP_THRESHOLD:
                    is_close_up = True

                feet_px = [int((x1 + x2) / 2), y2]
                x_m, y_m = transform_pixel_to_meters(H, feet_px)

                # Strict FIBA Half-Court Limits
                if -7.5 <= x_m <= 7.5 and -0.5 <= y_m <= 11.5:
                    player_count += 1
                    raw_frame_objects.append({
                        "track_id": int(track_id),
                        "class": "player",
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(round(conf, 2)),
                        "feet_coord_px": feet_px,
                        "court_coords_meters": [round(x_m, 3), round(y_m, 3)]
                    })

        # --- 3. ASSEMBLE TELEMETRY ---
        frame_data = {
            "frame_id": frame_idx,
            "is_valid_court_view": True,
            "cut_reason": None,
            "objects": raw_frame_objects
        }

        if is_close_up:
            frame_data["is_valid_court_view"] = False
            frame_data["cut_reason"] = "NON_STANDARD_ANGLE"
            frame_data["objects"] = []
        elif player_count < MIN_PLAYERS_REQUIRED:
            frame_data["is_valid_court_view"] = False
            frame_data["cut_reason"] = "INSUFFICIENT_PLAYERS"
            frame_data["objects"] = []

        tracking_payload["frames"].append(frame_data)
        frame_idx += 1

    cap.release()
    tracking_payload["metadata"]["total_frames"] = len(tracking_payload["frames"])

    with open(output_json_path, 'w') as f:
        json.dump(tracking_payload, f, indent=2)
    print(f" Master Telemetry safely serialized to: {output_json_path}")
# =========================================================
# DIAGNOSTIC EXECUTION BLOCK
# =========================================================
if __name__ == "__main__":
    # 1. Update these to match your EXACT file locations
    VIDEO_FILE = "Court_footage_test.mp4"
    CALIB_FILE = "calibration.json"
    TRACKNET_CKPT = "weights/TrackNet_best.pt"
    INPAINTNET_CKPT = "weights/InpaintNet_best.pt"

    # Diagnostic: Check if files actually exist
    paths_to_check = [VIDEO_FILE, CALIB_FILE, TRACKNET_CKPT, INPAINTNET_CKPT]
    missing_files = [p for p in paths_to_check if not os.path.exists(p)]

    if missing_files:
        print(f" ERROR: Missing files found: {missing_files}")
        print("Please ensure your working directory contains these files.")
    else:
        print(" All files found. Launching pipeline...")

        # 2. Add an explicit 'verbose' error output
        try:
            run_fused_pipeline(
                video_path=VIDEO_FILE,
                calibration_path=CALIB_FILE,
                output_json_path="fused_tracking_data.json",
                tracknet_ckpt=TRACKNET_CKPT,
                inpaintnet_ckpt=INPAINTNET_CKPT
            )
        except Exception as e:
            print(f" A Python-level error occurred: {e}")