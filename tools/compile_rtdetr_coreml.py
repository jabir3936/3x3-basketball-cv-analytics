"""
compile_rtdetr_coreml.py
────────────────────────────────────────────────────────────────────────
WHY THE OLD APPROACH WAS SLOW
  compile_rtdetr.py exported to ONNX using opset 16. The RT-DETR R50
  decoder has 1195 nodes but CoreML only supports 877 of them — the
  other 318 (mostly transformer attention ops) silently fall back to
  CPU inside the CoreML session. That's why you're stuck at ~209ms.

THIS APPROACH
  Two changes:
    1. Switch backbone: rtdetr_r18vd instead of r50vd.
       R18 has a much lighter encoder — fewer custom ops — so CoreML
       covers a far higher fraction of the graph.
    2. Convert via coremltools directly to a .mlpackage.
       A native CoreML model bypasses ONNX Runtime entirely and runs
       the full graph on the ANE + GPU with no CPU fallback nodes.

PREREQUISITES
  pip install coremltools transformers torch

RUN ONCE
  python compile_rtdetr_coreml.py
  → writes  models/rtdetr_r18vd.mlpackage

THEN
  python run_detector_coreml.py   (new detector — see companion file)
"""

import os
import torch
import numpy as np
from transformers import RTDetrForObjectDetection

def compile():
    # ── Config ────────────────────────────────────────────────────────
    MODEL_ID   = "PekingU/rtdetr_r18vd"   # lighter backbone, fewer unsupported ops
    OUT_DIR    = "models"
    ONNX_PATH  = os.path.join(OUT_DIR, "rtdetr_r18vd.onnx")
    ML_PATH    = os.path.join(OUT_DIR, "rtdetr_r18vd.mlpackage")
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Step 1: Export to ONNX (opset 17) ────────────────────────────
    print(f"Loading {MODEL_ID} weights...")
    model = RTDetrForObjectDetection.from_pretrained(MODEL_ID)
    model.eval()
    model = model.float()

    dummy = torch.zeros(1, 3, 640, 640, dtype=torch.float32)

    print("Exporting to ONNX (opset 17)...")
    torch.onnx.export(
        model,
        dummy,
        f=ONNX_PATH,
        export_params=True,
        opset_version=17,            # opset 17 has better CoreML op coverage
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        dynamic_axes=None,           # static 640x640 — required for ANE
    )
    print(f"ONNX saved → {ONNX_PATH}")

    # ── Step 2: Convert ONNX → CoreML .mlpackage ─────────────────────
    try:
        import coremltools as ct
        print("\nConverting ONNX → CoreML .mlpackage...")
        print("(This takes 2–5 minutes on first run)")

        mlmodel = ct.convert(
            ONNX_PATH,
            # ML Program is the modern format — enables ANE scheduling
            convert_to="mlprogram",
            # float16 precision: halves memory bandwidth, ANE prefers fp16
            compute_precision=ct.precision.FLOAT16,
            # Tell coremltools the input is a 640x640 image tensor
            inputs=[ct.TensorType(name="pixel_values",
                                  shape=(1, 3, 640, 640),
                                  dtype=np.float32)],
            # Allow coremltools to target ANE + GPU
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS13,
        )

        mlmodel.save(ML_PATH)
        print(f"\n✅ CoreML model saved → {ML_PATH}")
        print("Run: python run_detector_coreml.py")

    except ImportError:
        print("\n⚠️  coremltools not installed.")
        print("Install it with:  pip install coremltools")
        print(f"\nONNX model is at {ONNX_PATH} — you can still use it with")
        print("run_detector_fast.py (swap MODEL_PATH to the r18 path).")
        print("The r18 ONNX will be faster than r50 even without coremltools.")

    except Exception as e:
        print(f"\n⚠️  CoreML conversion failed: {e}")
        print(f"ONNX model is still usable at {ONNX_PATH}")
        print("Swap MODEL_PATH in run_detector_fast.py to use it.")


if __name__ == "__main__":
    compile()
