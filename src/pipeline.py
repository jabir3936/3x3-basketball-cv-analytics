import threading
import queue
import time
import cv2

class AsynchronousTrackingPipeline:
    def __init__(self, video_path, max_buffer_size=32):
        self.video_path = video_path
        self.max_buffer_size = max_buffer_size
        
        # Thread-safe queue to pass frames from the reader to the AI engine
        self.frame_queue = queue.Queue(maxsize=self.max_buffer_size)
        
        # Concurrency control signals
        self.stopped = False
        self.read_thread = None

    def start(self):
        """Spawns the background video decoding thread."""
        self.stopped = False
        self.read_thread = threading.Thread(target=self._frame_producer, daemon=True)
        self.read_thread.start()
        return self

    def _frame_producer(self):
        """Target worker loop: continuously populates the memory buffer."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Failed to open video source at {self.video_path}")
            self.stopped = True
            return

        while not self.stopped:
            # If the queue fills up, block here until the AI engine consumes a frame
            if not self.frame_queue.full():
                success, frame = cap.read()
                if not success:
                    # Video stream has reached its end
                    self.stopped = True
                    break
                
                self.frame_queue.put(frame)
            else:
                # Brief sleep to prevent high-frequency CPU spinning while waiting for slot
                time.sleep(0.001)

        cap.release()
        print("Background Video Producer Thread safely closed.")

    def read_frame(self):
        """Pulls the next available decoded video canvas from the queue."""
        try:
            # Short timeout prevents the main thread from freezing if the queue empties temporarily
            return self.frame_queue.get(timeout=0.1)
        except queue.Empty:
            return None

    def has_frames(self):
        """Checks if the video is still processing or frames remain in the pipeline."""
        return not self.stopped or not self.frame_queue.empty()

    def stop(self):
        """Gracefully tears down the worker configurations."""
        self.stopped = True
        if self.read_thread:
            self.read_thread.join()