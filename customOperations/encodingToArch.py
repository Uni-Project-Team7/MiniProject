import torch
import torch.nn as nn
from archBuilder import *

# 0	Nafnet
# 1	Lakd-Net
# 2	UFP-Deblur
# 3	CGNet
# 4	CaptNet
# 5	Restormer
# 6	Loformer
# 7	SwinIR
# 8	FFT-Former
# 9 conv-default

# input = (x, 3, 512, 512)
# after 1st down-sample = (x, 64, 256, 256)
# after enc0 = (x, 64, 256, 256)
# after 2nd down-sample = (x, 128, 128, 128)
# after enc1 = (x, 128, 128, 128)
# after 3rd down-sample = (x, 256, 64, 64)
# after enc2  = (x, 256, 64, 64)
# after bottleneck = (x, 256, 64, 64)
# after 1st up-sample = (x, 128, 128, 128)
# after dec0 = (x, 128, 128, 128)
# after 1st skip connection = (x, 256, 128, 128)
# after 2nd up-sample = (x, 128,
def decode_and_build_unet(model_array):
    # Parsing models
    dim = 64
    model1 = model_array[0]
    model2 = model_array[1]
    model3 = model_array[2]
    bottleneck_model = model_array[3]

    model1_params = [model_array[4], model_array[5], 0, dim]
    model2_params = [model_array[6], model_array[7], 1, dim]
    model3_params = [model_array[8], model_array[9], 2, dim]
    bottleneck_params = [model_array[10], model_array[11], 3, dim]

    stage0 = model_function(model1, model1_params)
    stage1 = model_function(model2, model2_params)
    stage2 = model_function(model3, model3_params)
    stage3 = model_function(bottleneck_model, bottleneck_params)

    # Constructing the UNet
    unet_model = build_unet(stage0, stage1, stage2, stage3, dim)
    return unet_model


# Example of a model function
def model_function(model_type, params):
    match model_type:
        case 0:
            return nafnet_builder(params)
        case 1:
            return lakd_builder(params)
        case 2:
            return UFP_builder(params)
        case 3:
            return CG_builder(params)
        case 4:
            return Capt_builder(params)
        case 5:
            return Rest_builder(params)
        case 6:
            return Lo_builder(params)
        case 7:
            return Swin_builder(params)
        case 8:
            return FFT_builder(params)
        case 9:
            return conv_def_builder(params)


# Example of a UNet builder
def build_unet(encoder1, encoder2, encoder3, bottleneck):
    # This function would assemble the UNet architecture using the provided encoders and bottleneck
    print("Assembling UNet with:")
    print(f"  Encoder 1: {encoder1}")
    print(f"  Encoder 2: {encoder2}")
    print(f"  Encoder 3: {encoder3}")
    print(f"  Bottleneck: {bottleneck}")
    # Here you'd implement the UNet construction logic
    return "UNet_model"


# Example usage
model_array = ["modelA", "modelB", "modelC", "modelD",
               0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

unet = decode_and_build_unet(model_array)
print("Final UNet model:", unet)


class Generator(nn.Module):
    def __init__(self, stage0, stage1, stage2, bottleneck, dim):
        super().__init__()

        self.initial = nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1)

        self.enc0 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1),
            stage0[0]
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=2, padding=1),
            stage1[0]
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 8, kernel_size=3, stride=2, padding=1),
            stage2[0]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(dim * 8, dim * 8, kernel_size=3, stride=2, padding=1),
            bottleneck[0]
        )
        self.dec0 = nn.Sequential(
            nn.Conv2d(dim * 8, dim * 8 * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            stage0[1]
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(dim * 4 * 2, dim * 4 * 2 * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            stage1[1]
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(dim * 2 * 2, dim * 2 * 2 * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            stage2[1]
        )

        self.final = nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x, test=False):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        bottle = self.bottleneck(x)
        d0 = self.dec0(bottle)
        d1 = self.dec1(torch.cat([d0, e2], 1))
        d2 = self.dec(torch.cat([d1, e1], 1))
        out = self.final(d2)
        if test:
            return e0, e1, e2, bottle, d0, d1, d2, out

        return out
