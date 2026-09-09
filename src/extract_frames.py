import os
import cv2
from ultralytics import YOLO # Changed from RTDETR to YOLO
import torch

def extract_and_split_frames(video_paths, output_base_dir, total_target_frames=1200, train_ratio=0.8):
    # Set up the proper Ultralytics directory structure
    train_dir = os.path.join(output_base_dir, "train")
    val_dir = os.path.join(output_base_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Detect device: MPS for Apple Silicon, CPU fallback for older Intel Macs
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🎬 Loading YOLOv8 Nano for fast smart filtering on device: {device}...")
    
    # FIX: Use YOLOv8 Nano. It's blazing fast on Mac and perfect for detecting people to check framing.
    model = YOLO("models/yolov8n.pt") 
    
    frames_per_video = total_target_frames // len(video_paths)
    total_extracted = 0
    
    for vid_idx, video_path in enumerate(video_paths):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open {video_path}")
            continue
            
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Robust fallback if OpenCV fails to read frame count (common with some MP4 encodings)
        if total_frames <= 0 or total_frames > 500000: 
            print(f"⚠️ Warning: Unreliable frame count for {video_path}. Using safe estimate.")
            total_frames = 100000 
            
        # Strict timeline split to prevent Data Leakage
        split_frame_index = int(total_frames * train_ratio)
        
        # Calculate target frame counts for this specific video
        train_target = int(frames_per_video * train_ratio)
        val_target = frames_per_video - train_target
        
        saved_train = 0
        saved_val = 0
        frame_idx = 0
        
        # Step size ensures we sample across the whole video, not just the first minute
        step_size = max(1, total_frames // (frames_per_video * 2))
        
        print(f"\n📼 Processing Video {vid_idx + 1}/{len(video_paths)}: {os.path.basename(video_path)}")
        print(f"🎯 Target: {train_target} Train | {val_target} Val | Step: {step_size}")
        
        # Sequential reading is exponentially faster than cap.set() inside a loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # End of video
                
            # Only process every Nth frame
            if frame_idx % step_size != 0:
                frame_idx += 1
                continue

            # --- THE STRICT SMART FILTER ---
            # classes=[0] ensures we ONLY look for 'person' (COCO class 0)
            results = model.predict(frame, classes=[0], conf=0.25, verbose=False, device=device)
            r = results[0]
            
            is_valid_wide_shot = True # Default: assume it's a valid wide shot
            
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                
                # 1. Check for extreme close-ups (players take up too much vertical space)
                max_player_height = max([box[3] - box[1] for box in boxes])
                if max_player_height > (orig_h * 0.30): # Lowered to 30% to be stricter
                    is_valid_wide_shot = False
                    
                # 2. Check for "Replay" framing (e.g., picture-in-picture, heavy cropping)
                total_player_area = sum([(box[2] - box[0]) * (box[3] - box[1]) for box in boxes])
                frame_area = orig_h * orig_w
                if frame_area > 0 and (total_player_area / frame_area) > 0.40: # >40% screen coverage = reject
                    is_valid_wide_shot = False
            
            # --- ROUTING LOGIC (Strict Timeline Split) ---
            if is_valid_wide_shot:
                filename = f"vid{vid_idx+1}_frame_{frame_idx:05d}.jpg"
                
                if frame_idx < split_frame_index:
                    if saved_train < train_target:
                        cv2.imwrite(os.path.join(train_dir, filename), frame)
                        saved_train += 1
                        total_extracted += 1
                else:
                    if saved_val < val_target:
                        cv2.imwrite(os.path.join(val_dir, filename), frame)
                        saved_val += 1
                        total_extracted += 1
                        
                # Progress indicator
                print(f"  ✅ Vid {vid_idx+1} -> Train: {saved_train}/{train_target} | Val: {saved_val}/{val_target}", end="\r")
                
            frame_idx += 1
            
            # Early exit if both targets are met for this video (saves massive processing time)
            if saved_train >= train_target and saved_val >= val_target:
                break
                
        cap.release()
        print(f"\n🏁 Finished {os.path.basename(video_path)}.")
        
    print(f"\n🎉 Extraction Complete! {total_extracted} valid frames cleanly split into '{output_base_dir}'.")

# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    video_files = [
        "media/footage/training/match_footage_1.mp4", 
        "media/footage/training/match_footage_2.mp4", 
        "media/footage/training/match_footage_3.mp4"
    ]
    
    extract_and_split_frames(
        video_paths=video_files, 
        output_base_dir="dataset/images", 
        total_target_frames=1200, 
        train_ratio=0.8
    )