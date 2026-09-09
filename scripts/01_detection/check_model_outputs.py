"""
check_model_outputs.py
Run this once to see exactly what your ONNX model outputs.
Place it in the same folder as run_detector_fast.py and run:
    python check_model_outputs.py
"""
import onnxruntime as ort
import numpy as np

MODEL_PATH = "models/rtdetr_r50vd_fixed32.onnx"

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

print("=== INPUTS ===")
for inp in session.get_inputs():
    print(f"  name={inp.name!r}  shape={inp.shape}  dtype={inp.type}")

print("\n=== OUTPUTS ===")
for out in session.get_outputs():
    print(f"  name={out.name!r}  shape={out.shape}  dtype={out.type}")

# Run a dummy forward pass and print actual output shapes
dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
input_name = session.get_inputs()[0].name
results = session.run(None, {input_name: dummy})

print("\n=== ACTUAL OUTPUT SHAPES (from dummy run) ===")
for i, r in enumerate(results):
    print(f"  output[{i}]  shape={r.shape}  dtype={r.dtype}")
