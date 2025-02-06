import numpy as np
#import torch
import pycuda.driver as cuda
#import pycuda.autoinit
import tensorrt as trt
import os

# NOTE: meant to test trt engine created via onnx dynamic, yet the exported onnx model does not support dynamic shape range
# This file could be though of as pseudo code for testing dynamic shape range in TensorRT

def load_engine(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

current_directory = os.getcwd()
print(f"Current working directory: {current_directory}")
engine = load_engine("/scripts/dinov2_dynamic_trt.engine")
context = engine.create_execution_context()

stream = cuda.Stream()
batch_size = 1
height = 518
width = np.random.randint(518, 1554) # Random test input with dynamic width within the new range [518, 1554]


input_shape = (batch_size, 3, height, 518)
output_shape = (batch_size, width, 768)
d_input = cuda.mem_alloc(np.prod(input_shape) * np.float32().itemsize)
d_output = cuda.mem_alloc(np.prod(output_shape) * np.float32().itemsize)


input_data = np.random.randn(batch_size, 3, height, width).astype(np.float32)

# Adjust context for the new width
context.set_binding_shape(0, (batch_size, 3, height, width))

# Transfer data to GPU
cuda.memcpy_htod(d_input, input_data)

# Run inference
bindings = [int(d_input), int(d_output)]
context.execute_v2(bindings)

# Copy the output back to CPU
output_data = np.empty(context.get_binding_shape(1), dtype=np.float32)
cuda.memcpy_dtoh(output_data, d_output)

print(f"Inference output shape: {output_data.shape}")
print("Inference completed successfully.")
