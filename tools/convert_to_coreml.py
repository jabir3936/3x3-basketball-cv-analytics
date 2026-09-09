"""
convert_to_coreml.py
────────────────────────────────────────────────────────────────────────
Converts RT-DETR r18 → native CoreML .mlpackage via torch.jit.trace.
This bypasses ONNX entirely — coremltools converts directly from PyTorch.

PREREQUISITES
  pip install coremltools transformers torch

RUN
  python convert_to_coreml.py
  → writes models/rtdetr_r18vd.mlpackage

THEN
  python run_detector_coreml.py
"""

import os
import numpy as np
import torch
import coremltools as ct
from transformers import RTDetrForObjectDetection

print(f"coremltools version: {ct.__version__}")

MODEL_ID = "PekingU/rtdetr_r18vd"
ML_PATH  = "models/rtdetr_r18vd.mlpackage"
os.makedirs("models", exist_ok=True)

# ── 1. Load model ─────────────────────────────────────────────────────────────
print(f"\nLoading {MODEL_ID}...")
model = RTDetrForObjectDetection.from_pretrained(MODEL_ID)
model.eval()
model = model.float()

# ── 2. Wrap model to return only logits + pred_boxes ─────────────────────────
# coremltools traces the full forward pass — we wrap it so only the two
# outputs we care about are part of the traced graph.
class RTDetrWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits, outputs.pred_boxes

wrapper = RTDetrWrapper(model)
wrapper.eval()

# ── 3. Trace with TorchScript ────────────────────────────────────────────────
print("Tracing model with torch.jit.trace (static 640×640)...")
dummy = torch.zeros(1, 3, 640, 640, dtype=torch.float32)

with torch.no_grad():
    traced = torch.jit.trace(wrapper, dummy)

print("Trace complete.")

# ── 4. Convert to CoreML .mlpackage ──────────────────────────────────────────
print("\nConverting to CoreML .mlpackage...")
print("(Takes 3–8 minutes — this is the slow step)")

mlmodel = ct.convert(
    traced,
    convert_to="mlprogram",                  # required for ANE
    compute_precision=ct.precision.FLOAT16,  # ANE prefers fp16
    inputs=[
        ct.TensorType(
            name="pixel_values",
            shape=(1, 3, 640, 640),
            dtype=np.float32,
        )
    ],
    outputs=[
        ct.TensorType(name="logits"),
        ct.TensorType(name="pred_boxes"),
    ],
    compute_units=ct.ComputeUnit.ALL,        # ANE + GPU + CPU
    minimum_deployment_target=ct.target.macOS13,
)

mlmodel.save(ML_PATH)
print(f"\n✅ Saved → {ML_PATH}")
print("Run:  python run_detector_coreml.py")
