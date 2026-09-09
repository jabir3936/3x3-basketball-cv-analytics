import json
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

class TrajectoryProcessor:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.frames = self.data.get("frames", [])
        
    def process_and_save(self, output_path, window_size=11, poly_order=3):
        print("Extracting trajectories...")
        trajectories = {}
        
        # 1. Extract
        for frame in self.frames:
            if not frame.get("is_valid_court_view", True):
                continue
            
            frame_id = frame["frame_id"]
            for obj in frame.get("objects", []):
                track_id = obj.get("track_id", "ball")
                coords = obj.get("court_coords_meters")
                
                if coords:
                    if track_id not in trajectories:
                        trajectories[track_id] = {"frames": [], "x": [], "y": []}
                    trajectories[track_id]["frames"].append(frame_id)
                    trajectories[track_id]["x"].append(coords[0])
                    trajectories[track_id]["y"].append(coords[1])
                    
        print("Interpolating and Smoothing...")
        smoothed_data = {}
        # 2. Interpolate & Smooth
        for track_id, track_data in trajectories.items():
            if len(track_data["frames"]) < window_size:
                continue
                
            f_interp_x = interp1d(track_data["frames"], track_data["x"], kind='linear', fill_value="extrapolate")
            f_interp_y = interp1d(track_data["frames"], track_data["y"], kind='linear', fill_value="extrapolate")
            
            full_frame_range = np.arange(track_data["frames"][0], track_data["frames"][-1] + 1)
            interp_x = f_interp_x(full_frame_range)
            interp_y = f_interp_y(full_frame_range)
            
            smooth_x = savgol_filter(interp_x, window_size, poly_order)
            smooth_y = savgol_filter(interp_y, window_size, poly_order)
            
            # Map back to frame IDs
            smoothed_data[track_id] = {
                f_id: [round(float(sx), 3), round(float(sy), 3)] 
                for f_id, sx, sy in zip(full_frame_range, smooth_x, smooth_y)
            }

        print("Injecting smoothed coordinates back into JSON payload...")
        # 3. Inject back into JSON
        for frame in self.frames:
            frame_id = frame["frame_id"]
            for obj in frame.get("objects", []):
                track_id = obj.get("track_id", "ball")
                
                # If we have smoothed data for this track at this frame, add it
                if track_id in smoothed_data and frame_id in smoothed_data[track_id]:
                    obj["smoothed_court_coords_meters"] = smoothed_data[track_id][frame_id]

        with open(output_path, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"✅ Smoothed JSON saved to {output_path}")

if __name__ == "__main__":
    # Point this to your week 2 JSON output
    processor = TrajectoryProcessor("data/tracking/archive/match_tracking_data10.08.json")
    processor.process_and_save("data/tracking/archive/match_tracking_data_smoothed.json")