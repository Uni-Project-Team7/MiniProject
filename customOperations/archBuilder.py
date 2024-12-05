from base_arch.LakdNet import Mixerblock as LakdBlock
from base_arch.fftFormer import TransformerBlock as FftBlock
import torch
import torch.nn as nn
def nafnet_builder(params):
    pass


def lakd_builder(params):
    dim = params[3] * (2 ** params[2])
    param1 = int(params[0])
    param2 = int(params[1] * 2 + 1)
    print(param1, param2, dim
          )
    if params[2] != 3:
        return [nn.Sequential(*[LakdBlock(dim=dim, mix_kernel_size=param2, ffn_expansion_factor=2.3,
                                           bias=False, LayerNorm_type='WithBias') for i in range(param1)]),
                nn.Sequential(*[LakdBlock(dim=dim, mix_kernel_size=param2, ffn_expansion_factor=2.3,
                                           bias=False, LayerNorm_type='WithBias') for i in range(param1)])
                ]
    else :
        return [nn.Sequential(*[LakdBlock(dim=dim, mix_kernel_size=param2, ffn_expansion_factor=2.3,
                                           bias=False, LayerNorm_type='WithBias') for i in range(param1)])]

def UFP_builder(params):
    pass


def CG_builder(params):
    pass


def Capt_builder(params):
    pass


def Rest_builder(params):
    pass


def Lo_builder(params):
    pass


def Swin_builder(params):
    pass


def FFT_builder(params):
    dim = params[3] * (2 ** params[2])
    param1 = int(params[0])
    print(param1, dim)
    if params[2] != 3:
        return [nn.Sequential(*[FftBlock(dim=dim, ffn_expansion_factor=3, bias=False) for i in range(param1)]),
                nn.Sequential(*[FftBlock(dim=dim, att=True, ffn_expansion_factor=3, bias=False) for i in range(param1)])
                ]
    else :
        return [nn.Sequential(*[FftBlock(dim=dim, att=True, ffn_expansion_factor=3, bias=False) for i in range(param1)])]


def conv_def_builder(params):
    dim = params[3] * (2 ** params[2])
    return [nn.Conv2d(in_channels = dim, out_channels = dim, kernel_size = 1), nn.Conv2d(in_channels = dim, out_channels = dim, kernel_size = 1)]


