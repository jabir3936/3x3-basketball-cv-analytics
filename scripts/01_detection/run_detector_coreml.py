"""
run_detector_coreml.py
────────────────────────────────────────────────────────────────────────
Uses the native .mlpackage produced by compile_rtdetr_coreml.py.
Bypasses ONNX Runtime entirely — CoreML schedules directly onto the
Apple Neural Engine + GPU with no CPU fallback nodes.

PREREQUISITE
  python compile_rtdetr_coreml.py   ← run once first

USAGE
  python run_detector_coreml.py

FALLBACK
  If coremltools isn't installed, this script falls back automatically
  to the ONNX path with the lighter r18 backbone (still faster than r50).
"""

import cv2
import time
import queue
import threading
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
VIDEO_PATH        = "media/footage/calibration_test.mov"
COREML_MODEL_PATH = "models/rtdetr_r18vd.mlpackage"   # native CoreML
ONNX_MODEL_PATH   = "models/rtdetr_r18vd.onnx"         # fallback
CONFIDENCE_THRESH = 0.45
INPUT_SIZE        = 640
DECODE_BUFFER     = 64
SHOW_WINDOW       = True


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ASYNC VIDEO READER  (unchanged from run_detector_fast.py)
# ─────────────────────────────────────────────────────────────────────────────
class AsyncVideoReader:
    def __init__(self, video_path: str, queue_size: int = DECODE_BUFFER):
        self.stream      = cv2.VideoCapture(video_path)
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.stopped     = False
        self.thread      = None
        if not self.stream.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        self.width      = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height     = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.native_fps = self.stream.get(cv2.CAP_PROP_FPS)

    def start(self):
        self.stopped = False
        self.thread  = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        return self

    def _update_loop(self):
        while not self.stopped:
            if not self.frame_queue.full():
                ok, frame = self.stream.read()
                if not ok:
                    self.stopped = True
                    break
                self.frame_queue.put(frame)
            else:
                time.sleep(0.001)
        self.stream.release()

    def read_frame(self):
        return self.frame_queue.get() if not self.frame_queue.empty() else None

    def is_running(self):
        return not self.stopped or not self.frame_queue.empty()

    def stop(self):
        self.stopped = True
        if self.thread:
            self.thread.join(timeout=2.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SHARED PREPROCESSING  (same for both backends)
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(frame: np.ndarray) -> tuple[np.ndarray, int, int]:
    orig_h, orig_w = frame.shape[:2]
    resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE),
                         interpolation=cv2.INTER_LINEAR)
    rgb    = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = (rgb.astype(np.float32) / 255.0)
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    return tensor, orig_w, orig_h


# ─────────────────────────────────────────────────────────────────────────────
# 3a. NATIVE COREML BACKEND  (.mlpackage via coremltools)
# ─────────────────────────────────────────────────────────────────────────────
class CoreMLBackend:
    def __init__(self, model_path: str):
        import coremltools as ct
        print(f"Loading native CoreML model: {model_path}")
        self.model = ct.models.MLModel(
            model_path,
            compute_units=ct.ComputeUnit.ALL,   # ANE + GPU + CPU
        )
        print("✅  Native CoreML loaded — full ANE/GPU scheduling active")
        self._warmup()

    def _warmup(self):
        print("Warming up (3 frames)...")
        dummy = {"pixel_values": np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE),
                                          dtype=np.float32)}
        for _ in range(3):
            self.model.predict(dummy)
        print("Warmup complete.\n")

    def infer(self, frame: np.ndarray):
        tensor, orig_w, orig_h = preprocess(frame)
        out = self.model.predict({"pixel_values": tensor})
        # coremltools returns a dict keyed by output name
        logits    = out["logits"][0]      # shape (300, 80)
        pred_boxes = out["pred_boxes"][0]  # shape (300, 4)
        return logits, pred_boxes, orig_w, orig_h


# ─────────────────────────────────────────────────────────────────────────────
# 3b. ONNX FALLBACK BACKEND  (r18 via onnxruntime + CoreML EP)
# ─────────────────────────────────────────────────────────────────────────────
class ONNXBackend:
    def __init__(self, model_path: str):
        import onnxruntime as ort
        print(f"Loading ONNX model (r18 fallback): {model_path}")
        coreml_opts = {
            "COREML_FLAG_USE_CPU_AND_GPU": "1",
            "COREML_FLAG_ONLY_ALLOW_STATIC_INPUT_SHAPES": "1",
        }
        providers = [("CoreMLExecutionProvider", coreml_opts),
                     "CPUExecutionProvider"]
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = \
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session    = ort.InferenceSession(model_path,
                                               sess_options=sess_opts,
                                               providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        active = self.session.get_providers()
        print(f"   Active providers: {active}")
        self._warmup()

    def _warmup(self):
        print("Warming up (3 frames)...")
        dummy = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        for _ in range(3):
            self.session.run(["logits", "pred_boxes"],
                             {self.input_name: dummy})
        print("Warmup complete.\n")

    def infer(self, frame: np.ndarray):
        tensor, orig_w, orig_h = preprocess(frame)
        logits, pred_boxes = self.session.run(
            ["logits", "pred_boxes"], {self.input_name: tensor}
        )
        return logits[0], pred_boxes[0], orig_w, orig_h


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DETECTION DRAWING
# ─────────────────────────────────────────────────────────────────────────────
CLASSES = {0: "Player", 32: "Ball"}
COLORS  = {0: (180, 105, 255), 32: (0, 165, 255)}

def draw_detections(frame, logits, pred_boxes, orig_w, orig_h, threshold):
    for i in range(len(logits)):
        scores   = logits[i]
        class_id = int(np.argmax(scores))
        conf     = float(scores[class_id])
        if conf < threshold or class_id not in CLASSES:
            continue
        cx, cy, w, h = pred_boxes[i]
        x1 = int((cx - w / 2) * orig_w)
        y1 = int((cy - h / 2) * orig_h)
        x2 = int((cx + w / 2) * orig_w)
        y2 = int((cy + h / 2) * orig_h)
        color = COLORS[class_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{CLASSES[class_id]} {conf:.2f}",
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# 5.  ROLLING FPS COUNTER
# ─────────────────────────────────────────────────────────────────────────────
class PerfOverlay:
    def __init__(self, window: int = 30):
        self._times = []
        self._window = window

    def tick(self, elapsed_ms: float):
        self._times.append(elapsed_ms)
        if len(self._times) > self._window:
            self._times.pop(0)

    @property
    def avg_ms(self):
        return sum(self._times) / len(self._times) if self._times else 0.0

    @property
    def fps(self):
        return 1000.0 / self.avg_ms if self.avg_ms > 0 else 0.0

    def draw(self, frame, backend_name):
        text = f"{backend_name}  |  {self.fps:.1f} FPS  |  {self.avg_ms:.1f} ms"
        cv2.rectangle(frame, (0, 0), (len(text) * 10 + 20, 40), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 80), 2)
        return frame


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import os

    # Pick the best available backend
    if os.path.exists(COREML_MODEL_PATH):
        try:
            engine = CoreMLBackend(COREML_MODEL_PATH)
            backend_name = "CoreML Native (ANE)"
        except ImportError:
            print("coremltools not installed — falling back to ONNX r18")
            engine = ONNXBackend(ONNX_MODEL_PATH)
            backend_name = "ONNX r18 + CoreML EP"
    elif os.path.exists(ONNX_MODEL_PATH):
        print("No .mlpackage found — using ONNX r18 fallback")
        engine = ONNXBackend(ONNX_MODEL_PATH)
        backend_name = "ONNX r18 + CoreML EP"
    else:
        raise FileNotFoundError(
            "No model found. Run compile_rtdetr_coreml.py first.\n"
            f"  Expected: {COREML_MODEL_PATH}\n"
            f"  Fallback: {ONNX_MODEL_PATH}"
        )

    reader  = AsyncVideoReader(VIDEO_PATH).start()
    overlay = PerfOverlay(window=30)

    print(f"Video: {VIDEO_PATH}  "
          f"({reader.width}×{reader.height} @ {reader.native_fps:.0f} FPS native)")
    print("Press 'q' to quit.\n")

    if SHOW_WINDOW:
        cv2.namedWindow("RT-DETR Fast", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RT-DETR Fast", 1280, 720)

    while reader.is_running():
        frame = reader.read_frame()
        if frame is None:
            time.sleep(0.002)
            continue

        t0 = time.perf_counter()
        logits, pred_boxes, orig_w, orig_h = engine.infer(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        overlay.tick(elapsed_ms)
        frame = draw_detections(frame, logits, pred_boxes,
                                orig_w, orig_h, CONFIDENCE_THRESH)
        frame = overlay.draw(frame, backend_name)

        if SHOW_WINDOW:
            cv2.imshow("RT-DETR Fast", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    reader.stop()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()
    print(f"\nDone.  {overlay.avg_ms:.1f} ms avg  |  {overlay.fps:.1f} FPS")


if __name__ == "__main__":
    main()
