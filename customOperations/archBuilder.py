from base_arch.LakdNet import Mixerblock
import torch
import torch.nn as nn
def nafnet_builder(params):
    pass


def lakd_builder(params):
    blocks = []


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
    pass


def conv_def_builder(params):
    dim = params[3] * (2 ** params[2])
    return [nn.Conv2d(in_channels = dim, out_channels = dim, kernel_size = 1), nn.Conv2d(in_channels = dim, out_channels = dim, kernel_size = 1)]


