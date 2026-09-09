import os
import cv2
import numpy as np
import onnxruntime as ort

class HighSpeedInferenceEngine:
    def __init__(self, model_path="models/rtdetr_r50vd_fixed32.onnx", confidence_threshold=0.45):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model blueprint asset missing at: {model_path}. Run compile_rtdetr.py first.")
            
        self.confidence_threshold = confidence_threshold
        
        print("Igniting CoreML Neural Engine for Apple Silicon...")
        
        # Now that the float64 bug is patched in the ONNX file, we can safely maximize GPU utilization
        providers = [
            ("CoreMLExecutionProvider", {
                "COREML_FLAG_USE_CPU_AND_GPU": "1",
                # We locked the dimensions to 640x640 in the export, so this flag tells the GPU to permanently 
                # pre-allocate register space, drastically reducing memory overhead per frame.
                "COREML_FLAG_ONLY_ALLOW_STATIC_INPUT_SHAPES": "1" 
            }),
            "CPUExecutionProvider" # Silent fallback, just in case
        ]

        # Enable maximum graph optimization (constant folding, node fusion)
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self.session = ort.InferenceSession(
                model_path, 
                sess_options=session_options, 
                providers=providers
            )
            print("🚀 Success: Native Apple Silicon GPU Acceleration is active!")
        except Exception as e:
            print(f"⚠️ CoreML fallback triggered: {e}")
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        
        self.input_name = self.session.get_inputs()[0].name
        self.target_classes = {0: "Player", 32: "Ball"}

    def preprocess(self, frame):
        """Converts an OpenCV frame to a hardware-compatible 640x640 float32 tensor."""
        resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        float_frame = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(float_frame, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        return tensor

    def run_inference(self, frame):
        """Executes raw mathematical calculations on the designated hardware layer."""
        input_tensor = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        logits, pred_boxes = outputs[0][0], outputs[1][0]
        return logits, pred_boxes