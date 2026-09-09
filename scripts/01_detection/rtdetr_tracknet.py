import os
import sys
import json
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from collections import deque
import gc
from ultralytics import RTDETR

# =========================================================
# WASB HELPER FUNCTIONS (Affine Transforms for HRNet)
# =========================================================
def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result

def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)
    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

# =========================================================
# NATIVE WASB TRACKER CLASS
# =========================================================
class WASBTracker:
    def __init__(self, weights_path, device='cuda', score_threshold_override=None, debug=True):
        # CRITICAL: Hydra requires the working directory to be the repo root
        os.chdir('/content/WASB-SBDT')
        sys.path.append('/content/WASB-SBDT/src')
        
        from hydra import compose, initialize_config_dir
        from models import build_model
        
        # FIX 1: The config folder is actually inside 'src/'
        config_dir = os.path.abspath("/content/WASB-SBDT/src/configs")
        
        # FIX 2: The primary config is 'eval', NOT 'root'
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            self.cfg = compose(
                config_name="eval", 
                overrides=[
                    "dataset=basketball", 
                    "model=wasb", 
                    f"detector.model_path={weights_path}"
                ]
            )
            
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cpu':
            print("⚠️ WARNING: WASB is designed for CUDA. CPU execution may fail or be extremely slow.")
            
        # Build and Load Model
        self.model = build_model(self.cfg)
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        # Handle different checkpoint formats safely
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Setup Parameters
        self.inp_wh = (self.cfg.model.inp_width, self.cfg.model.inp_height)
        self.frames_in = self.cfg.model.frames_in
        self.frame_buffer = deque(maxlen=self.frames_in)
        
        # Post-processing parameters from the official detector config
        self.score_threshold = self.cfg.detector.postprocessor.score_threshold
        self.use_hm_weight = self.cfg.detector.postprocessor.use_hm_weight

        # DIAGNOSTIC: let you override the (basketball-broadcast-tuned) threshold
        # without touching hydra overrides, and log the raw heatmap peak per
        # frame so you can see WHY the ball is/isn't clearing it.
        if score_threshold_override is not None:
            print(f"🎯 Overriding WASB score_threshold: {self.score_threshold} -> {score_threshold_override}")
            self.score_threshold = score_threshold_override
        self.debug = debug
        self.heatmap_scores = []

        # WASB localizes the ball; this state keeps one persistent ball identity.
        self.ball_track_id = 1
        self.last_ball_xy = None
        self.last_ball_velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.ball_missed = 0
        self.max_ball_gap = 5

        # Optional court-only ROI (pixel space). When set, predict() crops to
        # this box before the affine transform, so a small/distant ball isn't
        # squashed away when the full frame gets resized down to inp_wh.
        self.roi = None

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def set_court_roi(self, H, frame_w, frame_h, court_x=(-15, 15), court_y=(-8, 8), pad_px=60):
        """Project the known court boundary (in meters, via the inverse
        homography) back into pixel space and restrict WASB inference to
        that box. The pretrained checkpoint was trained on broadcast footage
        where the ball occupies a reasonable number of pixels; on a wide,
        static tactical-cam shot the ball can shrink to a handful of pixels
        after the full frame gets resized to the network's small input size.
        Cropping to the court first keeps more real pixels on the ball."""
        H_inv = np.linalg.inv(H)
        corners_m = [
            [court_x[0], court_y[0]], [court_x[1], court_y[0]],
            [court_x[1], court_y[1]], [court_x[0], court_y[1]],
        ]
        pts_px = []
        for mx, my in corners_m:
            vec = np.array([mx, my, 1.0], dtype=np.float64)
            p = H_inv @ vec
            pts_px.append([p[0] / p[2], p[1] / p[2]])
        pts_px = np.array(pts_px)
        x1 = max(int(pts_px[:, 0].min()) - pad_px, 0)
        y1 = max(int(pts_px[:, 1].min()) - pad_px, 0)
        x2 = min(int(pts_px[:, 0].max()) + pad_px, frame_w)
        y2 = min(int(pts_px[:, 1].max()) + pad_px, frame_h)
        self.roi = (x1, y1, x2, y2)
        print(f"🏀 WASB inference restricted to court ROI {self.roi} (full frame was {frame_w}x{frame_h})")

    def summarize(self):
        """Call after the video finishes to see whether low ball recall is a
        threshold/confidence problem (peaks cluster just under the line) or a
        deeper detection problem (peaks are low almost everywhere)."""
        if not self.heatmap_scores:
            print("[WASB debug] no frames were processed")
            return
        arr = np.array(self.heatmap_scores)
        print(
            f"[WASB debug] heatmap peak over {len(arr)} frames: "
            f"min={arr.min():.3f} median={np.percentile(arr,50):.3f} "
            f"p90={np.percentile(arr,90):.3f} max={arr.max():.3f} | "
            f"threshold={self.score_threshold} | "
            f"fraction of frames clearing it={(arr > self.score_threshold).mean():.1%}"
        )

    def predict(self, frame):
        x_off, y_off = 0, 0
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            frame = frame[y1:y2, x1:x2]
            x_off, y_off = x1, y1

        # Maintain temporal buffer (WASB expects 3 frames)
        self.frame_buffer.append(frame)
        while len(self.frame_buffer) < self.frames_in:
            self.frame_buffer.appendleft(frame)
            
        h, w = frame.shape[:2]
        center = np.array([w / 2., h / 2.], dtype=np.float32)
        scale = max(w, h) * 1.0
        trans = get_affine_transform(center, scale, 0, self.inp_wh)
        
        tensors = []
        for f in self.frame_buffer:
            inp_img = cv2.warpAffine(f, trans, self.inp_wh, flags=cv2.INTER_LINEAR)
            inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(inp_img))
            
        # Shape: (1, frames_in * 3, H, W)
        tensor_seq = torch.cat(tensors, dim=0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds = self.model(tensor_seq)
            
            # WASB HRNet returns a dict of scales, e.g., {0: tensor}
            if isinstance(preds, dict):
                scale_key = list(preds.keys())[0]
                hm_tensor = preds[scale_key].sigmoid()
            else:
                hm_tensor = preds.sigmoid()
                
            # Extract heatmap for the final frame in the sequence
            hm = hm_tensor[0, -1].cpu().numpy()
            
            # Explicitly clear VRAM
            del preds, hm_tensor, tensor_seq

        # DIAGNOSTIC: record the raw peak confidence regardless of whether it
        # clears the threshold, so summarize() can tell you which failure
        # mode you're in (see notes at the bottom of the file).
        self.heatmap_scores.append(float(np.max(hm)))
        if self.debug and len(self.heatmap_scores) % 200 == 0:
            recent = self.heatmap_scores[-200:]
            print(f"[WASB debug] frames {len(self.heatmap_scores)-199}-{len(self.heatmap_scores)}: "
                  f"heatmap peak min={min(recent):.3f} mean={sum(recent)/len(recent):.3f} "
                  f"max={max(recent):.3f} (threshold={self.score_threshold})")

        # Post-process: detect and maintain a persistent single-ball track.
        detection = self._detect_blob_concomp(hm)
        inv_trans = get_affine_transform(center, scale, 0, self.inp_wh, inv=1)

        if detection is not None:
            x, y, bbox_hm, confidence = detection
            orig_xy = affine_transform(np.array([x, y]), inv_trans)
            bx = int(orig_xy[0]) + x_off
            by = int(orig_xy[1]) + y_off

            p1 = affine_transform(np.array([bbox_hm[0], bbox_hm[1]]), inv_trans)
            p2 = affine_transform(np.array([bbox_hm[2], bbox_hm[3]]), inv_trans)
            bx1 = int(min(p1[0], p2[0])) + x_off
            by1 = int(min(p1[1], p2[1])) + y_off
            bx2 = int(max(p1[0], p2[0])) + x_off
            by2 = int(max(p1[1], p2[1])) + y_off

            min_size = 6
            if bx2 - bx1 < min_size:
                half = min_size // 2
                bx1, bx2 = bx - half, bx + half
            if by2 - by1 < min_size:
                half = min_size // 2
                by1, by2 = by - half, by + half

            new_xy = np.array([bx, by], dtype=np.float32)
            if self.last_ball_xy is not None:
                self.last_ball_velocity = new_xy - self.last_ball_xy
            self.last_ball_xy = new_xy
            self.ball_missed = 0

            return {
                "track_id": self.ball_track_id,
                "class": "ball",
                "bbox": [bx1, by1, bx2, by2],
                "confidence": round(float(confidence), 3),
                "center_coord_px": [bx, by],
                "predicted": False,
            }

        # Short-gap motion prediction prevents one weak WASB frame from
        # deleting the ball from the JSON stream.
        if self.last_ball_xy is not None and self.ball_missed < self.max_ball_gap:
            self.last_ball_xy = self.last_ball_xy + self.last_ball_velocity
            self.ball_missed += 1
            bx, by = map(int, np.round(self.last_ball_xy))
            half = 4
            return {
                "track_id": self.ball_track_id,
                "class": "ball",
                "bbox": [bx-half, by-half, bx+half, by+half],
                "confidence": 0.0,
                "center_coord_px": [bx, by],
                "predicted": True,
            }

        self.ball_missed += 1
        return None

    def _detect_blob_concomp(self, hm):
        """Return the strongest WASB ball blob as (x, y, bbox, confidence)."""
        peak = float(np.max(hm))
        if peak <= self.score_threshold:
            return None

        _, hm_th = cv2.threshold(hm, self.score_threshold, 1, cv2.THRESH_BINARY)
        n_labels, labels = cv2.connectedComponents(hm_th.astype(np.uint8))

        candidates = []
        for label in range(1, n_labels):
            ys, xs = np.where(labels == label)
            if len(xs) == 0:
                continue
            ws = hm[ys, xs]
            if self.use_hm_weight and float(ws.sum()) > 0:
                x = float(np.sum(xs * ws) / np.sum(ws))
                y = float(np.sum(ys * ws) / np.sum(ws))
                score = float(ws.sum())
            else:
                x = float(xs.mean())
                y = float(ys.mean())
                score = float(ws.mean())
            candidates.append((score, x, y, int(xs.min()), int(ys.min()),
                               int(xs.max()), int(ys.max())))

        if not candidates:
            return None

        score, x, y, x1, y1, x2, y2 = max(candidates, key=lambda c: c[0])
        return x, y, [x1, y1, x2, y2], peak

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
# FUSED EXECUTION PIPELINE (Native WASB + RT-DETR)
# ---------------------------------------------------------
def run_fused_pipeline(video_path, calibration_path, output_json_path, wasb_ckpt,
                        wasb_score_threshold=None, wasb_use_court_roi=True):
    print("🚀 Initializing Native WASB & RT-DETR Models...")
    wasb_tracker = WASBTracker(wasb_ckpt, device='cuda', score_threshold_override=wasb_score_threshold)
    os.chdir('/content')
    with open(calibration_path, 'r') as f:
        calib_data = json.load(f)
    H = np.array(calib_data["homography_matrix"], dtype=np.float32)
    player_model = RTDETR("models/rtdetr-l.pt")
    
    cap = cv2.VideoCapture(video_path)
    ORIG_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ORIG_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    CLOSE_UP_THRESHOLD = ORIG_H * 0.35
    MIN_PLAYERS_REQUIRED = 4

    if wasb_use_court_roi:
        wasb_tracker.set_court_roi(H, ORIG_W, ORIG_H)
    
    metadata = {
        "video_name": os.path.basename(video_path),
        "frame_width": ORIG_W, "frame_height": ORIG_H, "total_frames": TOTAL_FRAMES
    }
    
    print("📝 Streaming telemetry directly to disk to save RAM...")
    with open(output_json_path, 'w') as f_out:
        f_out.write('{\n  "metadata": ')
        json.dump(metadata, f_out)
        f_out.write(',\n  "frames": [\n')
        
        first_frame = True
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            raw_frame_objects = []
            
            # --- 1. NATIVE WASB BALL TRACKING ---
            ball_obj = wasb_tracker.predict(frame)
            if ball_obj is not None:
                bx, by = ball_obj["center_coord_px"]
                bx_m, by_m = transform_pixel_to_meters(H, (bx, by))
                ball_obj["court_coords_meters"] = [round(bx_m, 3), round(by_m, 3)]
                raw_frame_objects.append(ball_obj)
                    
            # --- 2. RT-DETR PLAYER TRACKING ---
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
                    if (y2 - y1) > CLOSE_UP_THRESHOLD: is_close_up = True
                        
                    feet_px = [int((x1 + x2) / 2), y2]
                    x_m, y_m = transform_pixel_to_meters(H, feet_px)
                    if -14.0 <= x_m <= 14.0 and -7.5 <= y_m <= 7.5:
                        player_count += 1
                        raw_frame_objects.append({
                            "track_id": int(track_id), "class": "player",
                            "bbox": [x1, y1, x2, y2], "confidence": float(round(conf, 2)),
                            "feet_coord_px": feet_px,
                            "court_coords_meters": [round(x_m, 3), round(y_m, 3)]
                        })
                del boxes, ids, confidences
            
            del results, r, frame
            if frame_idx % 50 == 0: gc.collect()
                
            frame_data = {
                "frame_id": frame_idx + 1, "is_valid_court_view": True, "cut_reason": None, "objects": raw_frame_objects
            }
            
            # FIX: Only filter out players during cutaways/close-ups, preserve the ball if found!
            if is_close_up:
                frame_data["is_valid_court_view"] = False
                frame_data["cut_reason"] = "NON_STANDARD_ANGLE"
                frame_data["objects"] = [obj for obj in raw_frame_objects if obj["class"] == "ball"]
            elif player_count < MIN_PLAYERS_REQUIRED:
                frame_data["is_valid_court_view"] = False
                frame_data["cut_reason"] = "INSUFFICIENT_PLAYERS"
                frame_data["objects"] = [obj for obj in raw_frame_objects if obj["class"] == "ball"]
                
            if not first_frame: f_out.write(',\n')
            f_out.write('    ' + json.dumps(frame_data))
            first_frame = False
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{TOTAL_FRAMES} frames...")
            
        f_out.write('\n  ]\n}')
    cap.release()
    wasb_tracker.summarize()
    print(f"✅ Master Telemetry safely serialized to: {output_json_path}")

# =========================================================
# DIAGNOSTIC EXECUTION BLOCK
# =========================================================
if __name__ == "__main__":
    # UPDATE THESE PATHS TO MATCH YOUR GOOGLE DRIVE
    VIDEO_FILE = "/content/Court_footage_test.mp4"
    CALIB_FILE = "/content/calibration.json"
    WASB_CKPT = "/content/model_weights/wasb_basketball_best.pth.tar"
    OUTPUT_JSON = "fused_tracking_data.json"

    paths_to_check = [VIDEO_FILE, CALIB_FILE, WASB_CKPT]
    missing_files = [p for p in paths_to_check if not os.path.exists(p)]
    
    if missing_files:
        print(f"❌ ERROR: Missing files found: {missing_files}")
    else:
        print("✅ All Colab paths verified. Launching Native Pipeline...")
        try:
            run_fused_pipeline(
                video_path=VIDEO_FILE, 
                calibration_path=CALIB_FILE,
                output_json_path=OUTPUT_JSON,
                wasb_ckpt=WASB_CKPT,
                wasb_score_threshold=0.20
            )
        except Exception as e:
            print(f"❌ A Python-level error occurred: {e}")