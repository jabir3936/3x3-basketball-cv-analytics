import cv2
import json
import os
import numpy as np

class StateDiagnosticVisualizer:
    def __init__(self, video_path, json_path, calibration_path):
        if not os.path.exists(video_path): raise FileNotFoundError(f"Missing video: {video_path}")
        if not os.path.exists(json_path): raise FileNotFoundError(f"Missing JSON: {json_path}")
        if not os.path.exists(calibration_path): raise FileNotFoundError(f"Missing calibration: {calibration_path}")
            
        self.video_path = video_path
        
        print("Loading tracking payload...")
        with open(json_path, 'r') as f:
            self.payload = json.load(f)
            
        print("Loading Calibration for Inverse Homography...")
        with open(calibration_path, 'r') as f:
            calib_data = json.load(f)
            
        # Get matrix and calculate its inverse
        self.H = np.array(calib_data["homography_matrix"], dtype=np.float32)
        self.H_inv = np.linalg.inv(self.H)
            
        self.frames_data = self.payload.get("frames", [])
        self.total_frames = len(self.frames_data)
        
        self.trajectory_history_raw = {}
        self.trajectory_history_smooth = {}

    def meters_to_pixels(self, x_m, y_m):
        """Projects real-world meters back to video camera pixels"""
        vec = np.array([x_m, y_m, 1.0], dtype=np.float32)
        transformed = np.dot(self.H_inv, vec)
        w = transformed[2]
        if w == 0: return (0, 0)
        return (int(transformed[0] / w), int(transformed[1] / w))

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps) if fps > 0 else 30
        
        window_name = "Smoothed vs Raw Diagnostics"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        frame_idx = 0
        is_paused = False
        
        while cap.isOpened() and frame_idx < self.total_frames:
            if not is_paused:
                ret, frame = cap.read()
                if not ret: break
                    
                frame_data = self.frames_data[frame_idx]
                
                if frame_data.get("is_valid_court_view", True):
                    for obj in frame_data.get("objects", []):
                        x1, y1, x2, y2 = obj["bbox"]
                        track_id = obj.get("track_id", "ball")
                        
                        # 1. Plot Raw Trajectory (Red)
                        center_x, center_y = int((x1 + x2) / 2), y2
                        if track_id not in self.trajectory_history_raw:
                            self.trajectory_history_raw[track_id] = []
                        self.trajectory_history_raw[track_id].append((center_x, center_y))
                        
                        if len(self.trajectory_history_raw[track_id]) > 30:
                            self.trajectory_history_raw[track_id].pop(0)
                            
                        pts_raw = self.trajectory_history_raw[track_id]
                        for i in range(1, len(pts_raw)):
                            cv2.line(frame, pts_raw[i-1], pts_raw[i], (0, 0, 255), 2) # RED for raw
                            
                        # 2. Plot Smoothed Trajectory (Green)
                        smoothed_coords = obj.get("smoothed_court_coords_meters")
                        if smoothed_coords:
                            px_x, px_y = self.meters_to_pixels(smoothed_coords[0], smoothed_coords[1])
                            
                            if track_id not in self.trajectory_history_smooth:
                                self.trajectory_history_smooth[track_id] = []
                            self.trajectory_history_smooth[track_id].append((px_x, px_y))
                            
                            if len(self.trajectory_history_smooth[track_id]) > 30:
                                self.trajectory_history_smooth[track_id].pop(0)
                                
                            pts_smooth = self.trajectory_history_smooth[track_id]
                            for i in range(1, len(pts_smooth)):
                                cv2.line(frame, pts_smooth[i-1], pts_smooth[i], (0, 255, 0), 4) # GREEN for smoothed

                        # Draw Box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

                cv2.imshow(window_name, frame)
                frame_idx += 1
                
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'): break
            elif key == ord(' '): is_paused = not is_paused

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ensure you are loading the NEW smoothed json and your calibration matrix
    visualizer = StateDiagnosticVisualizer(
        "media/footage/court_footage.mp4", 
        "data/tracking/archive/match_tracking_data_smoothed.json",
        "data/calibration/calibration_new.json" 
    )
    visualizer.run()