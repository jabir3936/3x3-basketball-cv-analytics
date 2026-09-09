import os
import torch
from transformers import RTDetrForObjectDetection

def compile_model_to_onnx():
    model_id = "PekingU/rtdetr_r50vd"
    output_dir = "models"
    output_path = os.path.join(output_dir, "rtdetr_r50vd_fixed32.onnx")

    # Ensure output directory exists in the root folder
    os.makedirs(output_dir, exist_ok=True)

    print(f"Fetching baseline Apache 2.0 weights for: {model_id}...")
    model = RTDetrForObjectDetection.from_pretrained(model_id)

    # 1. THE CRITICAL FIX: Force the entire model architecture into 32-bit standard precision.
    # This prevents the AIFI Transformer blocks from generating float64 Cosine matrices
    # that cause ARM NEON / ONNX Runtime hardware backend panics.
    print("Casting Transformer backbone and AIFI layers to strict float32 precision...")
    model.eval()
    model = model.float() 

    # 2. THE CRITICAL FIX: Ensure the input tensor explicitly declares a float32 dtype.
    print("Synthesizing fixed 640x640 float32 dummy inputs for the tracking graph...")
    dummy_pixel_values = torch.randn(1, 3, 640, 640, dtype=torch.float32)

    print("Beginning structural graph serialization (This may take a minute)...")
    
    try:
        torch.onnx.export(
            model,
            dummy_pixel_values,                  # The strict 32-bit tensor
            f=output_path,
            export_params=True,                  # Store the trained weights inside the model file
            opset_version=16,                    # Opset 16 contains the necessary Transformer operator support
            do_constant_folding=True,            # Bake static variables directly into the graph for speed
            input_names=['pixel_values'],
            output_names=['logits', 'pred_boxes'],
            dynamic_axes=None                    # STRICT DECOUPLING: No dynamic axes. We force a static 640x640 memory grid.
        )
        print(f"🚀 Success! Commercially safe float32 graph frozen at: {output_path}")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")

if __name__ == "__main__":
    compile_model_to_onnx()