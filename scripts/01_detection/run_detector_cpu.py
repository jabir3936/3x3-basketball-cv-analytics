import cv2
import torch
import numpy as np
import time
import queue
import threading
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

# ==============================================================================
# 1. ASYNCHRONOUS VIDEO READER (Prevents the video from lagging the model)
# ==============================================================================
class AsyncVideoReader:
    def __init__(self, video_path, queue_size=64):
        self.stream = cv2.VideoCapture(video_path)
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = None

    def start(self):
        self.stopped = False
        self.thread = threading.Thread(target=self._update_loop, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def _update_loop(self):
        while True:
            if self.stopped:
                break
            if not self.frame_queue.full():
                grabbed, frame = self.stream.read()
                if not grabbed:
                    self.stopped = True
                    break
                self.frame_queue.put(frame)
            else:
                time.sleep(0.001)
        self.stream.release()

    def read_frame(self):
        return self.frame_queue.get() if not self.frame_queue.empty() else None

    def is_running(self):
        return not self.frame_queue.empty() or not self.stopped

    def stop(self):
        self.stopped = True
        if self.thread:
            self.thread.join()

# ==============================================================================
# 2. STABLE RT-DETR ENGINE (Pure CPU Implementation)
# ==============================================================================
class StableRTDetrEngine:
    def __init__(self, model_id="PekingU/rtdetr_r50vd", confidence_threshold=0.45):
        # We are intentionally forcing the CPU to ensure 100% stability
        self.device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        
        print("Initializing Engine in Safe CPU Mode...")
        
        # Load the Hugging Face processors and weights
        self.processor = RTDetrImageProcessor.from_pretrained(model_id)
        self.model = RTDetrForObjectDetection.from_pretrained(model_id).to(self.device)
        
        # COCO mapping: 0 = Person, 32 = Sports Ball
        self.target_classes = {0: "Player", 32: "Ball"}

    def process_frame(self, frame):
        # Convert BGR to RGB for the transformer
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Prepare inputs and run inference
        inputs = self.processor(images=rgb_frame, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Rescale the output boxes back to the original video dimensions
        target_sizes = torch.tensor([frame.shape[:2]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
        )[0]

        annotated_frame = frame.copy()

        # Extract data locally
        scores = results["scores"].numpy()
        labels = results["labels"].numpy()
        boxes = results["boxes"].numpy()

        # Draw the bounding boxes
        for score, label, box in zip(scores, labels, boxes):
            class_id = int(label)
            
            if class_id in self.target_classes:
                x1, y1, x2, y2 = map(int, box)
                class_name = self.target_classes[class_id]
                
                # Orange for Ball, Purple for Players
                box_color = (0, 165, 255) if class_id == 32 else (180, 105, 255)
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                label_text = f"{class_name} {score:.2f}"
                cv2.putText(annotated_frame, label_text, (x1, y1 - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                
        return annotated_frame

# ==============================================================================
# 3. MAIN EXECUTION LOOP
# ==============================================================================
def main():
    # Replace with your actual video filename
    VIDEO_PATH = "media/footage/calibration_test.mov" 
    
    video_stream = AsyncVideoReader(VIDEO_PATH).start()
    detector = StableRTDetrEngine()

    print("\nStarting playback visualization stream. Press 'q' to stop.")
    cv2.namedWindow("Stable RT-DETR Test", cv2.WINDOW_NORMAL)

    while video_stream.is_running():
        frame = video_stream.read_frame()
        if frame is None:
            continue

        # Track execution time
        start_time = time.time()
        processed_canvas = detector.process_frame(frame)
        latency = (time.time() - start_time) * 1000 
        
        # Convert latency to FPS for the display overlay
        fps = 1000 / latency if latency > 0 else 0

        cv2.putText(processed_canvas, f"CPU Speed: {fps:.1f} FPS | Latency: {latency:.0f}ms", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Stable RT-DETR Test", processed_canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_stream.stop()
    cv2.destroyAllWindows()
    print("Run finished.")

if __name__ == "__main__":
    main()