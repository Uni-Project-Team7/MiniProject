import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

##########################################################################
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


####################################################
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


#############################
class Mixerlayer(nn.Module):
    def __init__(self, dim, mix_kernel_size, bias):
        super(Mixerlayer, self).__init__()

        self.dense_depth_1 = nn.Conv2d(dim, dim, kernel_size=mix_kernel_size, stride=1, padding=mix_kernel_size//2, groups=dim, bias=bias)

        self.dense_point_1 = nn.Conv2d(dim, dim,1)

        self.dense_depth_2 = nn.Conv2d(dim, dim, kernel_size=mix_kernel_size, stride=1, padding=mix_kernel_size//2, groups=dim, bias=bias)

        self.dense_point_2 = nn.Conv2d(dim, dim,1)

    def forward(self, x):

        mixer_step1 = F.gelu(self.dense_depth_1(x))+x
        mixer_step1 = x + F.gelu(self.dense_point_1(mixer_step1))

        mixer_step2 = F.gelu(self.dense_depth_2(mixer_step1))+x
        mixer_step2 = x + F.gelu(self.dense_point_2(mixer_step2))

        return mixer_step2

###############################################################################
class Mixerblock(nn.Module):
    def __init__(self, dim, mix_kernel_size, ffn_expansion_factor, bias, LayerNorm_type):
        super(Mixerblock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.mixer = Mixerlayer(dim, mix_kernel_size, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):

        x_src = x
        z0 = self.norm1(x)
        x = x + self.mixer(z0)
        x = x_src + self.ffn(self.norm2(x))

        return x
