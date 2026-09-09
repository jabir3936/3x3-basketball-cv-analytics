import cv2
import numpy as np
import time
import os

# Import our custom low-overhead modules
from src.engine import HighSpeedInferenceEngine
from src.pipeline import AsynchronousTrackingPipeline

# NOTE: If you have your BoT-SORT tracker library installed, import it here:
# from tracker.bot_sort import BoTSORT 

class SportsAnalyticsOrchestrator:
    def __init__(self, video_path, model_path="models/rtdetr_r50vd_fixed32.onnx"):
        print("====== Initializing Production Analytics Engine ======")
        
        # 1. Initialize our decoupled data layers
        self.pipeline = AsynchronousTrackingPipeline(video_path=video_path, max_buffer_size=32)
        self.engine = HighSpeedInferenceEngine(model_path=model_path, confidence_threshold=0.40)
        
        # 2. Mock or real tracking initialization (BoT-SORT fallback)
        # In production, swap this placeholder out with your actual BoT-SORT class
        print("Registering State Kalman Filters & BoT-SORT parameters...")
        
        # 3. Load your existing Homography Calibration Matrix
        # Replace this hardcoded matrix with your actual json loading code from tracker_engine.py
        print("Loading court homography projection matrix parameters...")
        self.H = np.array([
           [
      0.12790881307738752,
      -0.14795961076634162,
      -5.501858318645445
    ],
    [
      -0.053848363656815584,
      -0.31142717230093775,
      224.05882264060781
    ],
    [
      0.0005914181263193412,
      0.018862951703324096,
      1.0
    ]
        ]) # Initially had matrix

    def project_to_court(self, pixel_x, pixel_y):
        """Applies the homography dot product matrix to find real court coordinates."""
        pixel_point = np.array([pixel_x, pixel_y, 1.0]).reshape(3, 1)
        transformed = np.dot(self.H, pixel_point)
        
        # Normalize the homogenous coordinate vectors 
        scale_factor = transformed[2, 0]
        if scale_factor != 0:
            court_x = transformed[0, 0] / scale_factor
            court_y = transformed[1, 0] / scale_factor
            return court_x, court_y
        return None

    def start_processing_loop(self):
        # Fire up the background video reader core
        self.pipeline.start()
        time.sleep(1.0) # Allow the ring buffer to fill slightly before loading UI
        
        cv2.namedWindow("CoreML RT-DETR + Tracking Pipeline", cv2.WINDOW_NORMAL)
        print("\nPipeline Live. Executing asynchronous tracking loops. Press 'q' to stop.")

        while self.pipeline.has_frames():
            start_time = time.time()
            
            # Fetch the pre-loaded frame from RAM buffer
            frame = self.pipeline.read_frame()
            if frame is None:
                continue
                
            orig_h, orig_w = frame.shape[:2]

            # Execute CoreML inference pass over the target frame
            logits, pred_boxes = self.engine.run_inference(frame)

            # Process output data vectors
            for i in range(len(logits)):
                # Extract classification scores
                scores = logits[i]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Filter against our targeted metrics (0: Player, 32: Ball)
                if confidence > self.engine.confidence_threshold and class_id in self.engine.target_classes:
                    # RT-DETR outputs normalised boxes: [cx, cy, w, h]
                    cx, cy, w, h = pred_boxes[i]
                    
                    # Denormalize coordinates back to native video pixel scale
                    x1 = int((cx - w / 2) * orig_w)
                    y1 = int((cy - h / 2) * orig_h)
                    x2 = int((cx + w / 2) * orig_w)
                    y2 = int((cy + h / 2) * orig_h)
                    
                    # Calculate tracking anchor point: Bottom center of bounding box (feet position)
                    feet_x = int(cx * orig_w)
                    feet_y = y2
                    
                    # RUN SPATIAL INTELLIGENCE STEP: Map to real-world meters via Homography Matrix
                    court_coord = self.project_to_court(feet_x, feet_y)
                    
                    # Setup clear visual markers
                    class_label = self.engine.target_classes[class_id]
                    color = (0, 165, 255) if class_id == 32 else (180, 105, 255)
                    
                    # Render visual overlays onto canvas
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    if court_coord:
                        cx_m, cy_m = court_coord
                        cv2.putText(frame, f"{class_label} [{cx_m:.1f}m, {cy_m:.1f}m]", (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Calculate precise system latency performance metrics
            latency_ms = (time.time() - start_time) * 1000
            fps = 1000 / latency_ms if latency_ms > 0 else 0
            
            cv2.putText(frame, f"CoreML FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms", 
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow("CoreML RT-DETR + Tracking Pipeline", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Tear down all running threads safely
        self.pipeline.stop()
        cv2.destroyAllWindows()
        print(" System Performance Run Terminated Safely")

if __name__ == "__main__":
    VIDEO_PATH = "media/footage/calibration_test.mov" # Place your test video filename here
    orchestrator = SportsAnalyticsOrchestrator(video_path=VIDEO_PATH)
    orchestrator.start_processing_loop()