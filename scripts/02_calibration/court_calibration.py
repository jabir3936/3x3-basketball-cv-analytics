import cv2
import numpy as np
import json
import time
import threading
import queue

# ==============================================================================
# 1. ASYNCHRONOUS FRAME STREAMER (THE PRODUCER THREAD)
# ==============================================================================
class AsyncVideoReader:
    """
    Spawns a dedicated background thread to read video frames from disk 
    and push them into a thread-safe Queue. This eliminates disk I/O bottlenecks 
    from the main computer vision processing thread.
    """
    def __init__(self, video_path, queue_size=128):
        self.video_path = video_path
        self.stream = cv2.VideoCapture(video_path)
        
        # Thread-safe FIFO queue to hold decoded image frames
        self.frame_queue = queue.Queue(maxsize=queue_size)
        
        # Control flag to cleanly kill the background thread when processing stops
        self.stopped = False
        self.thread = None

    def start(self):
        """Starts the background frame producer thread."""
        self.stopped = False
        self.thread = threading.Thread(target=self._update_loop, args=())
        self.thread.daemon = True # Allows thread to exit when main program exits
        self.thread.start()
        return self

    def _update_loop(self):
        """Infinite loop executed solely by the background thread."""
        while True:
            if self.stopped:
                break

            if not self.frame_queue.full():
                grabbed, frame = self.stream.read()
                
                # --- FIXED HERE ---
                # Instead of calling self.stop(), just break out of the loop
                if not grabbed:
                    self.stopped = True
                    break
                    
                self.frame_queue.put(frame)
            else:
                time.sleep(0.001)

        # The stream is safely released when the loop breaks
        self.stream.release()

    def read_frame(self):
        """Called by your consumer process to pull the next pre-decoded frame."""
        if self.frame_queue.empty():
            return None
        return self.frame_queue.get()

    def is_running(self):
        """Checks if there are still frames left to process or read."""
        return not self.frame_queue.empty() or not self.stopped

    def stop(self):
        """Cleanly stops the thread and releases structures."""
        self.stopped = True
        if self.thread:
            self.thread.join()


# ==============================================================================
# 2. HOMOGRAPHY CALIBRATION SYSTEM
# ==============================================================================
class CourtCalibrator:
    def __init__(self):
        # Container to hold our 4 physical click locations from the screen
        self.clicked_pixel_points = []
        
        # Standard FIBA 3x3 Half-Court metric coordinates (Meters).
        # ORDER IS CRITICAL: Must exactly match your clockwise mouse click flow!
        # Center of the baseline directly under the hoop rim is treated as origin (0,0).
        self.real_world_court_meters = np.array([
            [-7.5, 11.0],  # Point 1: Top-Left (Half-court line / left sideline intersection)
            [7.5, 11.0],   # Point 2: Top-Right (Half-court line / right sideline intersection)
            [7.5, 0.0],    # Point 3: Bottom-Right (Baseline / right sideline intersection)
            [-7.5, 0.0]    # Point 4: Bottom-Left (Baseline / left sideline intersection)
        ], dtype=np.float32)
        
        self.homography_matrix = None

    def _mouse_click_callback(self, event, u, v, flags, param):
        """
        OpenCV UI interceptor function. Fires automatically whenever mouse activity 
        is detected inside our window container.
        """
        # Listen exclusively for independent Left-Click down triggers
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.clicked_pixel_points) < 4:
                self.clicked_pixel_points.append([u, v])
                print(f"Stored Anchor Point {len(self.clicked_pixel_points)}: Pixel coordinates ({u}, {v})")
                
                # Dynamically overlay visual feedback onto our interactive display canvas
                cv2.circle(param['display_image'], (u, v), 6, (0, 0, 255), -1)
                cv2.putText(param['display_image'], f"Pt {len(self.clicked_pixel_points)}", 
                            (u + 10, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(param['window_name'], param['display_image'])

    def run_interactive_calibration(self, first_frame):
        """Displays frame 0 and configures callback listener loops to extract anchors."""
        window_name = "Interactive Calibration Window - Click 4 Corners Clockwise (Top-Left First)"
        display_img = first_frame.copy()
        
        # Package image states securely so the isolated mouse thread context can alter them
        callback_params = {'display_image': display_img, 'window_name': window_name}
        
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_click_callback, callback_params)
        
        print("\n--- STANDBY FOR GEOMETRY CONFIGURATION ---")
        print("Click the 4 court boundaries in exact clockwise sequence:")
        print("1. Top-Left (Half-Court) -> 2. Top-Right (Half-Court) -> 3. Bottom-Right (Baseline) -> 4. Bottom-Left (Baseline)")
        
        # Keep window frozen until user completes all 4 clicks or hits 'q' to abort
        cv2.imshow(window_name, display_img)
        while len(self.clicked_pixel_points) < 4:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Calibration manually aborted.")
                cv2.destroyAllWindows()
                return False
                
        cv2.destroyWindow(window_name)
        
        # Convert user pixel inputs into an explicit floating-point matrix array
        pixel_array = np.array(self.clicked_pixel_points, dtype=np.float32)
        
        # Compute the 3x3 perspective homography matrix mapping pixels to real-world meters
        self.homography_matrix, status = cv2.findHomography(pixel_array, self.real_world_court_meters)
        print("\nSuccessfully Computed 3x3 Homography Matrix Transformation!")
        return True

    def save_calibration_to_disk(self, filename="data/calibration/calibration.json"):
        """Serializes matrix out to a lightweight static JSON asset."""
        if self.homography_matrix is not None:
            config_payload = {
                "homography_matrix": self.homography_matrix.tolist()
            }
            with open(filename, 'w') as f:
                json.dump(config_payload, f, indent=2)
            print(f"Persisted matrix payload safely to local disk asset: '{filename}'")


# ==============================================================================
# 3. COMPREHENSIVE RUNTIME BENCHMARK PIPELINEc
# ==============================================================================
def execute_pipeline(video_source, mode="asynchronous"):
    print(f"\n================ Running Engine In Mode: {mode.upper()} ================")
    
    # -------------------------------------------------------------
    # STEP A: Calibration & Initialization Configuration Phase
    # -------------------------------------------------------------
    temp_cap = cv2.VideoCapture(video_source)
    success, initial_frame = temp_cap.read()
    temp_cap.release()
    
    if not success:
        print("Error: Could not access video source file to pull frame 0.")
        return

    calibrator = CourtCalibrator()
    if not calibrator.run_interactive_calibration(initial_frame):
        return
    calibrator.save_calibration_to_disk()

    # -------------------------------------------------------------
    # STEP B: Core Processing Metric Evaluation Loop
    # -------------------------------------------------------------
    frame_count = 0
    start_time = time.time()

    if mode == "synchronous":
        # Standard block reader execution setup
        sync_stream = cv2.VideoCapture(video_source)
        while True:
            grabbed, frame = sync_stream.read()
            if not grabbed:
                break
            
            # --- This is where model inference and analytics would run ---
            frame_count += 1
            
        sync_stream.release()

    elif mode == "asynchronous":
        # Non-blocking, multi-threaded worker pipeline setup
        async_stream = AsyncVideoReader(video_source).start()
        
        # Process frames as long as background thread is alive or frames remain in queue
        while async_stream.is_running():
            frame = async_stream.read_frame()
            if frame is None:
                continue
                
            # --- This is where model inference and analytics would run ---
            frame_count += 1
            
        async_stream.stop()

    total_time = time.time() - start_time
    calculated_fps = frame_count / total_time
    print(f"Processed {frame_count} frames in {total_time:.2f} seconds. Runtime Speed: {calculated_fps:.2f} FPS")


if __name__ == "__main__":
    # Path configuration - replace this with your local 3x3 basketball video file path
    TARGET_VIDEO = "media/footage/court_footage.mp4"
    
    # To run this script right away without an actual video file, you can pass an integer 
    # value of '0' instead to initialize your laptop's integrated webcam for logic testing.
    # TARGET_VIDEO = 0

    # 1. First, test using standard serial tracking execution loops
    # execute_pipeline(TARGET_VIDEO, mode="synchronous")
    
    # 2. Comment out the synchronous line above and uncomment below to benchmark performance boosts
    execute_pipeline(TARGET_VIDEO, mode="asynchronous")