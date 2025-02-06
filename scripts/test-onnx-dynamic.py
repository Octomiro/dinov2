import onnxruntime as ort
import numpy as np
import torch


def test_onnx_model(onnx_model_path, patch_size, test_shapes):
    """
    Loads an ONNX model and tests it with multiple input shapes.

    :param onnx_model_path: Path to the ONNX model file
    :param patch_size: Patch size to ensure height and width are multiples
    :param test_shapes: List of (height, width) tuples to test
    """
    # Load ONNX model
    session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    for h, w in test_shapes:
        #assert h % patch_size == 0 and w % patch_size == 0, f"Height and width must be multiples of {patch_size}"
        if not((h % patch_size == 0) and (w % patch_size == 0)):
            raise Warning(f"Height and width must be multiples of {patch_size}, currently got {h} and {w}")
            continue
        # Create a random input tensor
        input_tensor = np.random.rand(1, 3, h, w).astype(np.float32)
        
        # Run inference
        outputs = session.run([output_name], {input_name: input_tensor})
        
        print(f"Tested shape: (1, 3, {h}, {w}) -> Output shape: {outputs[0].shape}")

if __name__ == "__main__":
    onnx_model_path = "dinov2_dynamic.onnx"  # Path to your ONNX model
    patch_size = 14  # Ensure height/width are multiples of this
    
    # Define test shapes (ensure they match TensorRT profile)
    test_shapes = [
        (518, 518),   # Minimum shape
        #(518, 742),   # Mid-range shape
        #(518, 1022),  # Maximum shape (as per TensorRT profile)
        (518, 518),
        (518, 518),
    ]

    # NOTE: Dynamic shape range is not supported by ONNXRuntime, it throws: "element_wise_ops.h:560 void onnxruntime::BroadcastIterator::Append(ptrdiff_t, ptrdiff_t) axis == 1 || axis == largest was false. Attempting to broadcast an axis by a dimension other than 1."
    
    test_onnx_model(onnx_model_path, patch_size, test_shapes)
