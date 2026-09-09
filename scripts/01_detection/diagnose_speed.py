"""
diagnose_speed.py
Run this to find out exactly why you're still getting low FPS.
    python diagnose_speed.py
"""
import time
import numpy as np
import onnxruntime as ort

MODEL_PATH = "models/rtdetr_r50vd_fixed32.onnx"

# ── 1. What providers are actually available on this machine? ──────────────
print("=" * 60)
print("1. AVAILABLE PROVIDERS ON THIS MACHINE")
print("=" * 60)
available = ort.get_available_providers()
for p in available:
    print(f"   {'✅' if p != 'CPUExecutionProvider' else '  '} {p}")

has_coreml = "CoreMLExecutionProvider" in available
print(f"\n   CoreML available: {'YES ✅' if has_coreml else 'NO ❌ — this is your problem'}")

# ── 2. Which provider actually runs the session? ───────────────────────────
print("\n" + "=" * 60)
print("2. WHICH PROVIDER IS ACTUALLY RUNNING")
print("=" * 60)

coreml_opts = {
    "COREML_FLAG_USE_CPU_AND_GPU": "1",
    "COREML_FLAG_ONLY_ALLOW_STATIC_INPUT_SHAPES": "1",
}
providers = [("CoreMLExecutionProvider", coreml_opts), "CPUExecutionProvider"]

sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(MODEL_PATH, sess_options=sess_opts, providers=providers)
active_providers = session.get_providers()
print(f"   Requested: {[p[0] if isinstance(p, tuple) else p for p in providers]}")
print(f"   Active:    {active_providers}")

if active_providers[0] == "CoreMLExecutionProvider":
    print("   CoreML IS active ✅")
else:
    print("   CoreML is NOT active ❌ — falling back to CPU")

# ── 3. Raw inference speed (no video, no drawing) ─────────────────────────
print("\n" + "=" * 60)
print("3. RAW INFERENCE SPEED (no video overhead)")
print("=" * 60)

dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
input_name = session.get_inputs()[0].name

# Warmup
print("   Warming up (5 frames)...")
for _ in range(5):
    session.run(["logits", "pred_boxes"], {input_name: dummy})

# Benchmark
print("   Benchmarking 20 frames...")
times = []
for _ in range(20):
    t0 = time.perf_counter()
    session.run(["logits", "pred_boxes"], {input_name: dummy})
    times.append((time.perf_counter() - t0) * 1000)

avg_ms = sum(times) / len(times)
print(f"\n   Average inference: {avg_ms:.1f} ms")
print(f"   Theoretical max:   {1000/avg_ms:.1f} FPS (inference only)")

# ── 4. CPU-only comparison ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. CPU-ONLY BASELINE (for comparison)")
print("=" * 60)

cpu_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
for _ in range(3):
    cpu_session.run(["logits", "pred_boxes"], {input_name: dummy})

cpu_times = []
for _ in range(10):
    t0 = time.perf_counter()
    cpu_session.run(["logits", "pred_boxes"], {input_name: dummy})
    cpu_times.append((time.perf_counter() - t0) * 1000)

cpu_avg = sum(cpu_times) / len(cpu_times)
print(f"   CPU average: {cpu_avg:.1f} ms  ({1000/cpu_avg:.1f} FPS)")

# ── 5. Summary & next steps ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. DIAGNOSIS SUMMARY")
print("=" * 60)

if not has_coreml:
    print("""
   ❌ CoreML provider not found in onnxruntime build.
   FIX: Uninstall your current onnxruntime and install the Apple Silicon build:

       pip uninstall onnxruntime onnxruntime-silicon -y
       pip install onnxruntime-silicon

   Then re-run this script.
""")
elif active_providers[0] != "CoreMLExecutionProvider":
    print("""
   ❌ CoreML is installed but not running the model.
   This usually means the model has unsupported ops.
   FIX: Re-export with opset 17 (see below).
""")
elif avg_ms > 150:
    print(f"""
   ⚠️  CoreML IS active but inference is still slow ({avg_ms:.0f} ms).
   This means the model is on CPU inside CoreML (not GPU/ANE).
   FIX: Re-export the ONNX model with opset 17 instead of 16
   and try the ANE-optimised session options below.
""")
else:
    speedup = cpu_avg / avg_ms
    print(f"""
   ✅ Everything looks good!
   CoreML speedup vs CPU: {speedup:.1f}x
   Inference alone: {avg_ms:.0f} ms ({1000/avg_ms:.0f} FPS)
   If end-to-end FPS is still low, the bottleneck is video decode or drawing.
""")
