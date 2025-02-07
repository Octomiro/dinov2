from torch2trt_dynamic import module2trt, BuildEngineConfig
import torch


from typing import Literal

MODEL_NAMES = {
    'giant': 'dinov2_vitg14',
    'large': 'dinov2_vitl14',
    'base': 'dinov2_vitb14',
    'small': 'dinov2_vits14',
}

dev = torch.device('cuda')

class Dino(torch.nn.Module):
    def __init__(self, model_size: Literal['small', 'base', 'large', 'giant']) -> None:
        super(Dino, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', MODEL_NAMES[model_size]).to(dev).eval()
        # to imply DINOv2 to onnx fixes, load the model from repo that includes commits in 'RRoundTable/dinov2/tree/dinov2-onnx'
        # Or simply load from current repo from (hubconf.dinov2_vitb14(for_onnx=True)))
        # self.model = torch.hub.load('RRoundTable/dinov2:dinov2-onnx', MODEL_NAMES[model_size]).to(dev).eval()
        
    def forward(self, x):
        return self.model.get_intermediate_layers(x)[0]
    

model = Dino(model_size='base') # Load and move model to CUDA

# create example data
x = torch.ones((1, 3, 518, 518)).cuda()

# convert to TensorRT feeding sample data as input
config = BuildEngineConfig(
        shape_ranges=dict(
            x=dict(
                min=(1, 3, 518, 518),
                opt=(2, 3, 518, 742),
                max=(4, 3, 518, 1022),
            )
        ))
trt_model = module2trt(
        model,
        args=[x],
        config=config)

x = torch.rand(1, 3, 518, 518).cuda()
with torch.no_grad():
    y = model(x)
    y_trt = trt_model(x)

# check the output against PyTorch
torch.testing.assert_close(y, y_trt)