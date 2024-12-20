from base_arch.LakdNet import Mixerblock as LakdBlock
from base_arch.FftFormer import TransformerBlock as FftBlock
from base_arch.Loformer import TransformerBlock_2b as LoformerBlock
from base_arch.Nafnet import NAFBlock
from base_arch.CGNet import NAFBlock0 as CGNafnet, CascadedGazeBlock as CGBlock
from base_arch.Restormer import TransformerBlock as RestormerBlock
from base_arch.CaptNet import TransformerBlock as CaptBlock, CNNBlock as CaptCNNBlock
import torch
import torch.nn as nn

def nafnet_builder(params):
    dim = params[3] * (2 ** params[2])
    param1 = int(params[0])
    param2 = int(params[1] * 2 + 1)
    print(param1, param2, dim)
    if params[2]!=3 :
        return[
            nn.Sequential(
                *[NAFBlock(16) for _ in range(param1)]),
            nn.Sequential(
                *[NAFBlock(16) for _ in range(param1)])
             ]
    else :
        return [
            nn.Sequential(
                *[NAFBlock(16) for _ in range(1)]
            )
        ]


def lakd_builder(params):
    dim = params[3] * (2 ** params[2])
    param1 = int(params[0])
    param2 = int(params[1] * 2 + 1)
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
    dim = params[3] * (2 ** params[2])
    param1 = int(params[0])
    param2 = int(params[1] * 2 + 1)
    if params[2] != 3:
        return [nn.Sequential(CGBlock(dim, GCE_Conv=param2), *[CGNafnet(dim) for i in range(param1 - 1)]),
                nn.Sequential(*[CGNafnet(dim) for i in range(param1)])]
    else :
        return [nn.Sequential(*[CGNafnet(dim) for i in range(param1)])]



def Capt_builder(params):
    dim = params[3] * (2 ** params[2])
    param1 = int(params[0])
    param2 = int(params[1] * 2 + 1)
    if params[2] != 3:
        return [nn.Sequential(*([CaptCNNBlock(dim) for _ in range(param1)] if params[2] <= 1
        else [CaptBlock(dim, num_heads=param2) for _ in range(param1)])),
                nn.Sequential(*[CGNafnet(dim) for i in range(param1)])]
    else :
        return [nn.Sequential(*[CGNafnet(dim) for i in range(param1)])]



def Rest_builder(params):
    dim = params[3] * (2 ** params[2])
    param1 = int(params[0])
    param2 = int(params[1] * 2 + 1)
    if params[2] != 3:
        return [nn.Sequential(*[RestormerBlock(dim=dim, num_heads=param2, ffn_expansion_factor=2.66, bias=False,
                LayerNorm_type='WithBias') for i in range(param1)]),
                nn.Sequential(*[RestormerBlock(dim=dim, num_heads=param2, ffn_expansion_factor=2.66, bias=False,
                                               LayerNorm_type='WithBias') for i in range(param1)])]
    else :
        return [nn.Sequential(*[RestormerBlock(dim=dim, num_heads=param2, ffn_expansion_factor=2.66, bias=False,
                LayerNorm_type='WithBias') for i in range(param1)])]


def Swin_builder(params):
    pass

def Lo_builder(params):
    dim = params[3] * (2 ** params[2])
    param1 = int(params[0])
    param2 = int(params[1] * 2 + 1)
    print(param1, param2, dim)
    if params[2] != 3:
        return [
            nn.Sequential(
            *[LoformerBlock(dim=dim, num_heads=param2, ffn_expansion_factor=2.66,
                            bias=True, LayerNorm_type='WithBias',
                            window_size=8, window_size_dct=8,
                            num_k=8,
                            cs=['channel_mlp', 'channel_mlp'], norm_type_=[['LayerNorm', 'LayerNorm'], ['LayerNorm', 'LayerNorm']],
                            qk_norm=[False, False], temp_adj=None,
                            i=None, ffn='ffn') for _ in range(param1)]),
            nn.Sequential(
            *[LoformerBlock(dim=dim, num_heads=param2, ffn_expansion_factor=2.66,
                            bias=True, LayerNorm_type='WithBias',
                            window_size=8, window_size_dct=8,
                            num_k=8,
                            cs=['channel_mlp', 'channel_mlp'],
                            norm_type_=[['LayerNorm', 'LayerNorm'], ['LayerNorm', 'LayerNorm']],
                            qk_norm=[False, False], temp_adj=None,
                            i=None, ffn='ffn') for _ in range(param1)])
        ]

    else :
        return [nn.Sequential(
                *[LoformerBlock(dim=dim, num_heads=param2, ffn_expansion_factor=2.66,
                                bias=True, LayerNorm_type='WithBias',
                                window_size=8, window_size_dct=8,
                                num_k=8,
                                cs=['channel_mlp', 'channel_mlp'], norm_type_=[['LayerNorm', 'LayerNorm'], ['LayerNorm', 'LayerNorm']],
                                qk_norm=[False, False], temp_adj=None,
                                i=None, ffn='ffn') for _ in range(param1)]
            )]


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


