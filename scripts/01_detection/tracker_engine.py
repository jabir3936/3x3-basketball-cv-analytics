import cv2
import torch
import numpy as np
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

# Assuming you drop an open-source BoT-SORT implementation into a 'trackers' folder
# or use a permissive tracking library wrapper. 
# from trackers.bot_sort import BoTSORT 

class BroadcastTrackingEngine:
    def __init__(self, model_id="PekingU/rtdetr_r50vd", confidence_threshold=0.5):
        """
        Initializes the RT-DETR transformer and the BoT-SORT Re-ID tracker.
        """
        # Automatically map to GPU Tensor Cores if available, otherwise fallback to CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = confidence_threshold
        
        print(f"Initializing Engine on {self.device.upper()}...")

        # =====================================================================
        # 1. INITIALIZE RT-DETR (The Object Detector)
        # =====================================================================
        # The ImageProcessor handles the complex resizing and padding required by Vision Transformers
        self.processor = RTDetrImageProcessor.from_pretrained(model_id)
        
        # Load the raw model weights into VRAM
        self.model = RTDetrForObjectDetection.from_pretrained(model_id).to(self.device)
        
        # COCO Dataset mapping: We only care about ID 0 (Person) and ID 32 (Sports Ball)
        # We drop all other detections (chairs, cars, cups) to save processing time
        self.target_classes = [0, 32]

        # =====================================================================
        # 2. INITIALIZE BoT-SORT (The Multi-Object Tracker)
        # =====================================================================
        # We configure the tracker to handle broadcast camera cuts and heavy occlusions.
        # Note: This is pseudo-initialization based on standard BoT-SORT parameters.
        """
        self.tracker = BoTSORT(
            track_high_thresh=0.6,   # First Association: Trust detections above 60%
            track_low_thresh=0.1,    # Second Association: Recover blurry detections down to 10%
            new_track_thresh=0.7,    # Only create a brand new ID if we are 70% sure it's a person
            track_buffer=60,         # Keep IDs alive for 60 frames (2 seconds) if the camera cuts away
            match_thresh=0.8,        # Cosine similarity threshold for Re-ID appearance matching
            cmc_method="sparseOptFlow" # Camera Motion Compensation to handle broadcast panning
        )
        """

    def extract_detections(self, frame):
        """
        Takes a raw BGR OpenCV frame, runs it through RT-DETR, and returns 
        clean, filtered bounding boxes formatted for the tracker.
        """
        # Convert OpenCV BGR to standard RGB for the neural network
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pre-process the image into a PyTorch tensor
        inputs = self.processor(images=rgb_frame, return_tensors="pt").to(self.device)

        # Run the forward pass through the Vision Transformer (No gradients needed)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process the raw output logits into scaled bounding box coordinates
        target_sizes = torch.tensor([frame.shape[:2]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
        )[0]

        # Format the outputs into an array for BoT-SORT: [x1, y1, x2, y2, score, class_id]
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            class_id = label.item()
            
            # Filter out bench clutter: only keep players and the ball
            if class_id in self.target_classes:
                # Convert tensor to standard Python list of floats
                box_coords = box.cpu().numpy().tolist()
                detection_array = [
                    box_coords[0], # x_min
                    box_coords[1], # y_min
                    box_coords[2], # x_max
                    box_coords[3], # y_max
                    score.item(),  # Confidence Score
                    class_id       # 0 or 32
                ]
                detections.append(detection_array)

        return np.array(detections)

    def process_frame(self, frame):
        """
        The main loop. Ingests a frame, detects objects, and updates tracking IDs.
        """
        # 1. Get raw detections from the Transformer
        raw_detections = self.extract_detections(frame)
        
        # 2. Feed detections into BoT-SORT to update trajectories and handle ID assignments
        # (Assuming self.tracker.update() returns an array of active tracks)
        '''
        if len(raw_detections) > 0:
            tracked_objects = self.tracker.update(raw_detections, frame)
        else:
            tracked_objects = np.empty((0, 5))
            
        return tracked_objects
        '''
        
        # For now, just return the raw detections so we can verify RT-DETR works
        return raw_detections

# ==============================================================================
# Execution Block for Testing
# ==============================================================================
if __name__ == "__main__":
    # Initialize the engine
    engine = BroadcastTrackingEngine()
    
    # Load a test frame from your dataset
    test_frame = cv2.imread("Image.png")
    
    if test_frame is not None:
        outputs = engine.process_frame(test_frame)
        print("\nPipeline Output (x_min, y_min, x_max, y_max, confidence, class_id):")
        print(outputs)
    else:
        print("Please provide a valid test image to run the pipeline.")