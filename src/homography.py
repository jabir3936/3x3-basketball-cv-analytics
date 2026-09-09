import json
import os
import numpy as np

class SpatialTransformer:
    def __init__(self, calibration_path):
        if not os.path.exists(calibration_path):
            raise FileNotFoundError(f"Missing calibration matrix at: {calibration_path}")
            
        with open(calibration_path, 'r') as f:
            calib_data = json.load(f)
            
        # Load the matrix as a high-precision 32-bit float array for fast numpy operations
        self.H = np.array(calib_data["homography_matrix"], dtype=np.float32)
        print("3x3 Homography Matrix successfully loaded into memory.")

    def transform_point(self, pixel_point):
        """Converts a [x, y] pixel coordinate into [X, Y] meters."""
        x, y = pixel_point
        # Create the homogeneous coordinate vector [x, y, 1]
        vec = np.array([x, y, 1.0], dtype=np.float32)
        
        # Apply the dot product transformation
        transformed = np.dot(self.H, vec)
        
        # Divide by the perspective scaling factor (W)
        w = transformed[2]
        if w == 0: 
            return [0.0, 0.0] # Failsafe for mathematically parallel anomalies
            
        x_meters = float(transformed[0] / w)
        y_meters = float(transformed[1] / w)
        
        return [round(x_meters, 3), round(y_meters, 3)]

    def process_tracking_payload(self, input_json_path, output_json_path):
        """Iterates through the entire match timeline and projects all coordinates."""
        print(f"Ingesting raw pixel telemetry from: {input_json_path}")
        with open(input_json_path, 'r') as f:
            match_data = json.load(f)
            
        frames = match_data.get("frames", [])
        total_objects = 0
        
        for frame in frames:
            for obj in frame.get("objects", []):
                # We project the feet for players, and the center for the ball
                if obj["class"] == "player" and "feet_coord_px" in obj:
                    metric_coords = self.transform_point(obj["feet_coord_px"])
                    obj["court_coords_meters"] = metric_coords
                    total_objects += 1
                    
                elif obj["class"] == "ball" and "center_coord_px" in obj:
                    metric_coords = self.transform_point(obj["center_coord_px"])
                    obj["court_coords_meters"] = metric_coords
                    total_objects += 1
                    
        # Save the new metric-enriched payload
        with open(output_json_path, 'w') as f:
            json.dump(match_data, f, indent=2)
            
        print(f" Success: {total_objects} objects projected into 2D metric space.")
        print(f" Metric blueprint saved to: {output_json_path}")


if __name__ == "__main__":
    # Define your paths
    MATRIX_FILE = "data/calibration/calibration.json"
    RAW_TRACKING_FILE = "data/tracking/archive/match_tracking_data_full.json"
    METRIC_TRACKING_FILE = "data/tracking/archive/metric_tracking_data_old.json"
    
    try:
        transformer = SpatialTransformer(MATRIX_FILE)
        transformer.process_tracking_payload(RAW_TRACKING_FILE, METRIC_TRACKING_FILE)
    except Exception as e:
        print(f" Transformation failed: {e}")