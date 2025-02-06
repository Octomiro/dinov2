"""DINOV2 model converter to onnx."""
import torch
import argparse
import os
import sys
from pathlib import Path
current_path = Path(__file__).resolve()
parent_path = current_path.parent.parent.as_posix()
sys.path.insert(0, parent_path)
import hubconf
import tensorrt as trt



class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tensor):
        ff = self.model(tensor)
        return ff

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="dinov2_vits14", help="dinov2 model name")
parser.add_argument(
    "--image_height", type=int, default=518, help="input image height, must be a multiple of patch_size"
)
parser.add_argument(
    "--image_width", type=int, default=518, help="input image height, must be a multiple of patch_size"
)
parser.add_argument(
    "--patch_size", type=int, default=14, help="dinov2 model patch size, default is 16"
)
args = parser.parse_args()


if __name__ == "__main__":

    assert args.image_height % args.patch_size == 0, f"image height must be multiple of {args.patch_size}, but got {args.image_height}"
    assert args.image_width % args.patch_size == 0, f"image width must be multiple of {args.patch_size}, but got {args.image_height}"

    model = Wrapper(hubconf.dinov2_vits14(for_onnx=True)).to("cpu")
    model.eval()

    dummy_input = torch.rand([1, 3, args.image_height, args.image_width]).to("cpu")
    dummy_output = model(dummy_input)

    torch.onnx.export(
    model,
    #scripted_model,
    dummy_input,
    "dinov2_dynamic.onnx",
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    training=torch.onnx.TrainingMode.EVAL,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 3: "width"} # Dynamic batch size and width in input
        #"input": {3: "width"} # Dynamic width only
        #"input": {0: "batch_size", 2: "height", 3: "width"}
    }
)

logger = trt.Logger(trt.Logger.VERBOSE)

onnx_model_path = "dinov2_dynamic.onnx"


## NOTE: onnx to tensorRT engine, failed attempt 1
# Create builder and network
#with open(onnx_model_path, "rb") as f, trt.Builder(logger) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, logger) as parser:
#    
#    # Parse ONNX model
#    if not parser.parse(f.read()):
#        print("Failed to parse the ONNX file.")
#        for error in range(parser.num_errors):
#            print(parser.get_error(error))
#        raise ValueError("ONNX parsing failed.")
#
#    # Create optimization profile for dynamic shapes
#    config = builder.create_builder_config()
#    profile = builder.create_optimization_profile()
#    
#    # Specify dynamic shape range for input with fixed height and dynamic width
#    profile.set_dynamic_shape_profile("input", min=(1, 3, 518, 518), opt=(1, 3, 518, 518), max=(1, 3, 518, 1022)) # 1022, 1554
#
#    config.add_optimization_profile(profile)
#    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
#
#
#    # Build tensorRT engine
#    #engine = builder.build_engine(network, config)
#    engine = builder.build_serialized_network(network, config)
#
#    # Save the engine to a file
#    with open("dinov2_dynamic_trt.engine", "wb") as engine_file:
#        engine_file.write(engine.serialize())
