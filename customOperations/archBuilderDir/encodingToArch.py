import torch
import torch.nn as nn
from archBuilder import *
from archBuilder import Rest_builder

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

# enc0 : (x, 64, 512, 512)
# enc1 : (x, 128, 256, 256)
# enc2 : (x, 256, 128, 128)
# bottleneck : (x, 512, 64, 64)
# dec0 : (x, 256, 128, 128)
# dec1 : (x, 128, 256, 256)
# dec2 : (x, 64, 512, 512)

def decode_and_build_unet(model_array, dim = 64):
    # Parsing models
    model1 = model_array[0]
    model2 = model_array[1]
    model3 = model_array[2]
    bottleneck_model = model_array[3]
    #print("ia ma ")
    
    #print(model_array)
    model1_params = [model_array[4], model_array[5], 0, dim]
    model2_params = [model_array[6], model_array[7], 1, dim]
    model3_params = [model_array[8], model_array[9], 2, dim]
    bottleneck_params = [model_array[10], model_array[11], 3, dim]

    stage0 = model_function(model1, model1_params)
    stage1 = model_function(model2, model2_params)
    stage2 = model_function(model3, model3_params)
    stage3 = model_function(bottleneck_model, bottleneck_params)

    unet_model = Model(stage0, stage1, stage2, stage3, dim)
    return unet_model


def model_function(model_type, params):
    match model_type:
        case 0:
            return nafnet_builder(params)
        case 1:
            return lakd_builder(params)
        case 2:
            return FFT_builder(params)
        case 3:
            return CG_builder(params)
        case 4:
            return Rest_builder(params)
        case 5:
            return Capt_builder(params)
        case 6:
            return UFP_builder(params)
        case 7:
            return Swin_builder(params)
        case 8:
            return Lo_builder(params)
        case 9:
            return conv_def_builder(params)


class Downsample(nn.Module):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Model(nn.Module):
    def __init__(self, stage0, stage1, stage2, stage3, dim=64):
        super().__init__()
        self.initial = nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1)
        self.enc0 = stage0[0]
        self.down0 = Downsample(dim)
        self.enc1 = stage1[0]
        self.down1 = Downsample(dim * 2)
        self.enc2 = stage2[0]
        self.down2 = Downsample(dim * 4)
        self.bottleneck = stage3[0]
        self.up2 = Upsample(dim * 8)
        self.reduce2 = nn.Conv2d(in_channels=dim * 8, out_channels=dim * 4, kernel_size=1, bias=False)

        self.dec2 = stage2[1]
        self.up1 = Upsample(dim * 4)
        self.reduce1 = nn.Conv2d(in_channels=dim * 4, out_channels=dim * 2, kernel_size=1, bias=False)

        self.dec1 = stage1[1]

        self.up0 = Upsample(dim * 2)
        self.reduce0 = nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=1, bias=False)
        self.dec0 = stage0[1]

        self.final = nn.Conv2d(in_channels=dim, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x, test = False):
        embedded_tensor = self.initial(x)
        enc0_out = self.enc0(embedded_tensor)

        down0_out = self.down0(enc0_out)
        enc1_out = self.enc1(down0_out)

        down1_out = self.down1(enc1_out)
        enc2_out = self.enc2(down1_out)

        down2_out = self.down2(enc2_out)

        bottle = self.bottleneck(down2_out)

        up2_out = self.up2(bottle)
        skip2 = torch.cat([up2_out, enc2_out], 1)
        reduce_chan2 = self.reduce2(skip2)
        dec2_out = self.dec2(reduce_chan2)

        up1_out = self.up1(dec2_out)
        skip1 = torch.cat([up1_out, enc1_out], 1)
        reduce_chan1 = self.reduce1(skip1)
        dec1_out = self.dec1(reduce_chan1)

        up0_out = self.up0(dec1_out)
        skip0 = torch.cat([up0_out, enc0_out], 1)
        reduce_chan0 = self.reduce0(skip0)
        dec0_out = self.dec0(reduce_chan0)
        
        if test:
            return enc0_out, enc1_out, enc2_out, bottle, dec2_out, dec1_out, dec0_out, self.final(dec0_out)

        return self.final(dec0_out)
