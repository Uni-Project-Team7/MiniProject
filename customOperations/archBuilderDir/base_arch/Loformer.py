import torch
from torch import nn as nn
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F
import math
from torch._jit_internal import Optional
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.init import xavier_uniform_, constant
import kornia
import os
import numbers
from torch.nn import Sequential as Seq, Linear as Lin, Conv2d
from torch.nn.init import xavier_uniform_, constant_
from torch.nn import init as init

# !pip install kornia

class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):

        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        h, w = x.shape[-2:]
        x = check_image_size(x, self.patch_size)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x[:, :, :h, :w]).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
class DCTFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):

        super(DCTFFN, self).__init__()
        self.dct = DCT2x()
        self.idct = IDCT2x()
        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.quant = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        h, w = x.shape[-2:]
        x = check_image_size(x, self.patch_size)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_dct = self.dct(x_patch)
        x_patch_dct = x_patch_dct * self.quant
        x_patch = self.idct(x_patch_dct)
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x[:, :, :h, :w]).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, ffn='ffn', window_size=None):
        super(FeedForward, self).__init__()

        self.ffn_expansion_factor = ffn_expansion_factor

        self.ffn = ffn
        if self.ffn_expansion_factor == 0:
            hidden_features = dim
            self.project_in = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

            self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                                    groups=dim, bias=bias)
        else:
            hidden_features = int(dim*ffn_expansion_factor)
            self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
            self.act = nn.GELU()
            self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.dim = dim
        self.hidden_dim = hidden_features
    def forward(self, inp):
        x = self.project_in(inp)
        if self.ffn_expansion_factor == 0:
            x = self.act(self.dwconv(x))
        else:
            x1, x2 = self.dwconv(x).chunk(2, dim=1)
            x = self.act(x1) * x2
        x = self.project_out(x)
        return x


class GEGLU(nn.Module):
    def __init__(self, dim, kernel_size, bias):
        super(GEGLU, self).__init__()

        self.project_in = nn.Sequential(
            nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=kernel_size, padding=1, groups=dim * 2, bias=bias)
        )

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, inp):
        x = self.project_in(inp)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8, window_size_dct=9,
                 qk_norm=True, proj_out=True, temp_div=True, norm_dim=-1, cs='channel', padding_mode='zeros'):
        super().__init__()

        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.window_size_dct = window_size_dct
        self.grid_size = grid_size
        self.cs = cs
        # print(self.qk_norm)
        self.add = True if 'mlp_add' in self.cs else False
        self.channel_mlp = True if 'clp' in self.cs else False
        self.block_mlp = True if 'mlp' in self.cs else False
        self.coarse_mlp = True if 'coarse' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        self.global_attn = True if 'global' in self.cs else False
        if not self.global_attn:
            if 'grid' in self.cs:
                N = grid_size ** 2
                self.k = grid_size
            else:
                N = window_size ** 2
                self.k = window_size

        if self.block_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(N, N, bias=True),
                nn.GELU(),
            )
        if self.channel_mlp:
            self.cmlp = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1, bias=True),
                nn.GELU(),
            )
        # elif self.block_graph:
        #     self.graph = Grapher(dim, window_size=self.k)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()

    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        qkv = check_image_size(qkv, self.window_size)
        Hx, Wx = qkv.shape[-2:]
        if 'grid' in self.cs:
            qkv = rearrange(qkv, 'b (z head c) (h h1) (w w1) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                            h=self.grid_size, w=self.grid_size)
        else:
            qkv = rearrange(qkv, 'b (z head c) (h1 h) (w1 w) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                            h=self.window_size, w=self.window_size)
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        if self.block_mlp:
            if self.add:
                out = out + self.mlp(v)
            else:
                out = out * self.mlp(v)

        if 'grid' in self.cs:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h h1) (w w1)', head=self.num_heads, h1=Hx//self.grid_size,
                            w1=Wx//self.grid_size, h=self.grid_size, w=self.grid_size)
        else:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h1=Hx//self.window_size,
                            w1=Wx//self.window_size, h=self.window_size, w=self.window_size)
        if self.channel_mlp:
            if 'grid' in self.cs:
                v = rearrange(v, '(b h1 w1) head c (h w) -> b (head c) (h h1) (w w1)', head=self.num_heads,
                                h1=Hx // self.grid_size,
                                w1=Wx // self.grid_size, h=self.grid_size, w=self.grid_size)
            else:
                v = rearrange(v, '(b h1 w1) head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads,
                                h1=Hx // self.window_size,
                                w1=Wx // self.window_size, h=self.window_size, w=self.window_size)
            out = out * self.cmlp(v)
        return out[:, :, :H, :W]

    def get_attn_global(self, qkv):
        H, W = qkv.shape[-2:]
        qkv = rearrange(qkv, 'b (z head c) h w -> z b head c (h w)', z=3, head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        if self.block_mlp:
            out = out * self.mlp(v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        return out
    def forward(self, x):

        qkv = self.qkv_dwconv(self.qkv(x))
        # _, _, H, W = qkv.shape
        if not self.global_attn:
            out = self.get_attn(qkv)
        else:
            out = self.get_attn_global(qkv)
        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim=32,
                 num_heads=1,
                 ffn_expansion_factor=1,
                 bias=False,
                 LayerNorm_type='WithBias',
                 window_size=8,
                 window_size_dct=8,
                 num_k=8,
                 shift_size=0,
                 cs='channel',
                 norm_type=['LayerNorm', 'LayerNorm'],
                 qk_norm=False,
                 temp_adj=None,
                 ffn='ffn',
                 i=None):
        # def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        # print(window_size_dct)
        self.window_size_dct = window_size_dct
        # print(window_size_dct)
        # self.out_dir = i

        self.dim = dim
        self.num_k = num_k
        self.window_size = window_size
        self.shift_size = shift_size
        self.cs = cs

        self.window_size_dct = window_size_dct
        temp_div = True  # if not qk_norm else False

        if 'FLOPs' in cs:
            pass
        elif 'nodct' in cs:
            self.dct = nn.Identity()
            self.idct = nn.Identity()
        elif 'dct_torch' in cs:
            self.dct = DCT2x_torch()
            self.idct = IDCT2x_torch()
        else:
            self.dct = DCT2x()
            self.idct = IDCT2x()

        # self.attn = Attention_inter_wsca(dim, num_heads, bias, window_size=window_size, shift_size=shift_size, sca=False)
        if cs != 'identity':
            if norm_type[0] == 'InstanceNorm':
                norm1 = nn.InstanceNorm2d(dim)
            elif norm_type[0] == 'LayerNorm':
                norm1 = LayerNorm(dim, LayerNorm_type) # , out_dir=i)
            elif norm_type[0] == 'LayerNorm2x':
                norm1 = LayerNorm2x(dim, LayerNorm_type)
            elif norm_type[0] == 'LayerNorm2':
                norm1 = LayerNorm(dim*2, LayerNorm_type)
            elif norm_type[0] == 'LayerNorm_mu_sigma':
                norm1 = LayerNorm(dim, LayerNorm_type, True)
            elif norm_type[0] == 'BatchNorm':
                norm1 = nn.BatchNorm2d(dim)
            elif norm_type[0] == 'Softmax':
                norm1 = nn.Softmax(dim=1)
            else:
                norm1 = nn.Identity()
        else:
            norm1 = nn.Identity()
        if norm_type[1] == 'InstanceNorm':
            norm2 = nn.InstanceNorm2d(dim)
        elif norm_type[1] == 'LayerNorm':
            norm2 = LayerNorm(dim, LayerNorm_type)
        elif norm_type[1] == 'LayerNorm2':
            norm2 = LayerNorm(dim*2, LayerNorm_type)
        elif norm_type[1] == 'LayerNorm_mu_sigma':
            norm1 = LayerNorm(dim, LayerNorm_type, True)
        elif norm_type[1] == 'BatchNorm':
            norm2 = nn.BatchNorm2d(dim)
        else:
            norm2 = nn.Identity()
        self.norm1 = norm1

        self.attn = nn.Sequential(
            Attention(dim, num_heads, bias, window_size_dct=window_size_dct,
                         window_size=window_size, grid_size=num_k,
                         temp_div=temp_div, norm_dim=-1, qk_norm=qk_norm,
                         cs=cs, proj_out=True)
        )
        # self.attn = nn.Sequential(
        #     ProAttention(dim, num_heads, bias, window_size_dct=window_size_dct,
        #                   window_size=window_size, grid_size=num_k,
        #                   temp_div=temp_div, norm_dim=-1, qk_norm=qk_norm,
        #                   cs=cs, proj_out=True)
        # )
        self.norm2 = norm2
        if ffn == 'DFFN':
            self.ffn = nn.Sequential(
                DFFN(dim, ffn_expansion_factor, bias)
            )
        elif ffn == 'DCTFFN':
            self.ffn = nn.Sequential(
                DCTFFN(dim, ffn_expansion_factor, bias)
            )
        else:
            self.ffn = nn.Sequential(
                FeedForward(dim, ffn_expansion_factor, bias, ffn=ffn)
            )
        self.ffn_type = ffn

    def forward(self, x):
        # if 'nodct' in self.cs:
        #     x = self.attn(self.norm1(x)) + x
        # else:
        if 'LN_DCT' in self.cs:
            x_dct = self.dct(self.norm1(x))
            x_attn = self.attn(x_dct)
            x = self.idct(x_attn) + x
        else:
            x_dct = self.dct(x)
            x_attn = self.attn(self.norm1(x_dct))
            x_dct = x_dct + x_attn
            x = self.idct(x_dct)

        x_norm2 = self.norm2(x)
        x = x + self.ffn(x_norm2)
        return x

class TransformerBlock_2b(nn.Module):
    def __init__(self, dim=32,
                 num_heads=1,
                 ffn_expansion_factor=1,
                 bias=False,
                 LayerNorm_type='WithBias',
                 window_size=8,
                 window_size_dct=8,
                 num_k=8,
                 norm_type_=[['LayerNorm', 'LayerNorm'], ['LayerNorm', 'LayerNorm']],
                 qk_norm=[False, False],
                 cs=['channel', 'channel'],
                 temp_adj=None,
                 i=None,
                 ffn='ffn'):
        super().__init__()
        # print(window_size_dct)
        window_size_dct1 = None if window_size_dct < 1 else window_size_dct
        window_size_dct2 = None if window_size_dct < 1 else window_size_dct
        #     shift_size_ = [0, 0] # window_size_dct // 2] # [0, window_size_dct // 2]
        # else:
        #     window_size_dct1, window_size_dct2 = None, None
        shift_size_ = [0, 0]  # [0, window_size_dct // 2] # [0, 0]
        # print(cs, norm_type_, qk_norm)
        self.trans1 = TransformerBlock(dim, num_heads, ffn_expansion_factor,
                                       bias, LayerNorm_type, window_size, window_size_dct1, num_k=num_k,
                                       shift_size=shift_size_[0], cs=cs[0], norm_type=norm_type_[0],
                                       qk_norm=qk_norm[0], temp_adj=temp_adj, ffn=ffn
                                       )
        self.trans2 = TransformerBlock(dim, num_heads, ffn_expansion_factor,
                                       bias, LayerNorm_type, window_size, window_size_dct2, num_k=num_k,
                                       shift_size=shift_size_[1], cs=cs[1], norm_type=norm_type_[1],
                                       qk_norm=qk_norm[1], temp_adj=temp_adj, ffn=ffn
                                       )
        # self.conv_b = conv_bench(dim)
    def forward(self, x):
        x = self.trans1(x)
        x = self.trans2(x)

        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        # self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
        #                           nn.PixelUnshuffle(2))
    #     torch 1.7.1
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        # return self.body(x)

        return rearrange(self.body(x), 'b c (h h1) (w w1) -> b (c h1 w1) h w', h1=2, w1=2)
    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0
        flops += H * W * C * (C//2) * (3 * 3 + 1)
        return flops
class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0

        flops += H * W * C * (C * 2) * (3 * 3 + 1)
        return flops
def check_image_size(x, padder_size, mode='reflect'):
    _, _, h, w = x.size()
    mod_pad_h = (padder_size - h % padder_size) % padder_size
    mod_pad_w = (padder_size - w % padder_size) % padder_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode=mode)
    return x

def dct(x, W=None, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    # print(v.shape)
    Vc = torch.fft.fft(v, dim=1) # , onesided=False)
    # print(Vc.shape)
    if W is None:
        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)
    else:
        W_r, W_i = W
        W_r = W_r.to(x.device)
        W_i = W_i.to(x.device)
    V = Vc.real * W_r - Vc.imag * W_i # [:, :N // 2 + 1]

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)
    # print(V)
    return V

def idct(X, W=None, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2
    # print(X)
    if W is None:
        k = - torch.arange(N, dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)
    else:
        W_r, W_i = W
        W_r = W_r.to(X.device)
        W_i = W_i.to(X.device)
    # k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    # W_r = torch.cos(k)
    # W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    # V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    V = torch.complex(V_r, V_i)
    v = torch.fft.ifft(V, dim=1) # , onesided=False)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]
    x = x.real
    return x.view(*x_shape)


def dct_2d_torch(x, N_h=None, N_w=None, norm=None):

    X1 = dct(x, N_w, norm=norm)
    X2 = dct(X1.transpose(-1, -2), N_h, norm=norm)
    return X2.transpose(-1, -2)


def idct_2d_torch(X, N_h=None, N_w=None, norm=None):

    x1 = idct(X, N_w, norm=norm)
    x2 = idct(x1.transpose(-1, -2), N_h, norm=norm)
    return x2.transpose(-1, -2)

def get_dctMatrix(m, n):
    N = n
    C_temp = np.zeros([m, n])
    C_temp[0, :] = 1 * np.sqrt(1 / N)

    for i in range(1, m):
        for j in range(n):
            C_temp[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * N)) * np.sqrt(2 / N)
    return torch.tensor(C_temp, dtype=torch.float)


def dct1d(feature, dctMat):
    feature = feature @ dctMat.T # dctMat @ feature  #
    return feature.contiguous()  # torch.tensor(x, device=feature.device)


def idct1d(feature, dctMat):
    feature = feature @ dctMat # .T # dctMat.T @ feature  # .T
    return feature.contiguous()  # torch.tensor(x, device=feature.device)


def dct2d(feature, dctMat):
    # print(dctMat.shape, feature.shape)
    feature = dct1d(feature, dctMat)# dctMat @ feature
    # print(dctMat.shape, feature.shape)
    # feature = feature @ dctMat.T
    # print(feature.transpose(-1, -2).shape, dctMat.shape)
    feature = dct1d(feature.transpose(-1, -2), dctMat) # dctMat @ feature.transpose(-1, -2) # @ dctMat.T
    return feature.transpose(-1, -2).contiguous()  # torch.tensor(x, device=feature.device)


def idct2d(feature, dctMat):
    feature = idct1d(feature, dctMat) # dctMat.T @ feature # .transpose(-1, -2)
    feature = idct1d(feature.transpose(-1, -2), dctMat)
    return feature.transpose(-1, -2).contiguous() # torch.tensor(x, device=feature.device)

def dct2dx(feature, dctMat1, dctMat2):
    # print(dctMat.shape, feature.shape)
    feature = dct1d(feature, dctMat1) # dctMat1 @ feature
    # print(dctMat.shape, feature.shape)
    feature = dct1d(feature.transpose(-1, -2), dctMat2) # feature @ dctMat2.T
    return feature.transpose(-1, -2).contiguous()  # torch.tensor(x, device=feature.device)


def idct2dx(feature, dctMat1, dctMat2):
    feature = idct1d(feature, dctMat1)  # dctMat.T @ feature # .transpose(-1, -2)
    feature = idct1d(feature.transpose(-1, -2), dctMat2)
    return feature.transpose(-1, -2).contiguous()  # torch.tensor(x, device=feature.device)

class DCT1d(nn.Module):
    def __init__(self, window_size=64):
        super(DCT1d, self).__init__()
        self.dctMat = get_dctMatrix(window_size, window_size)

    def forward(self, x):
        self.dctMat = self.dctMat.to(x.device)
        # print(x.shape, self.dctMat.shape)
        x = dct1d(x, self.dctMat)
        return x


class IDCT1d(nn.Module):
    def __init__(self, window_size=64):
        super(IDCT1d, self).__init__()
        self.dctMat = get_dctMatrix(window_size, window_size)

    def forward(self, x):
        self.dctMat = self.dctMat.to(x.device)
        x = idct1d(x, self.dctMat)
        return x
class DCT1x(nn.Module):
    def __init__(self, dim=-1):
        super(DCT1x, self).__init__()
        self.dctMat = None
        self.dim = dim

    def check_dct_matrix(self, d):
        if self.dctMat is None or d != self.dctMat.shape[-1]:
            self.dctMat = get_dctMatrix(d, d)

    def forward(self, x):
        if self.dim != -1 or self.dim != len(x.shape)-1:
            x = x.transpose(self.dim, -1)
        self.check_dct_matrix(x.shape[-1])

        self.dctMat = self.dctMat.to(x.device)
        # print(x.shape, self.dctMat.shape)
        x = dct1d(x, self.dctMat)
        if self.dim != -1 or self.dim != len(x.shape)-1:
            x = x.transpose(self.dim, -1)
        return x.contiguous()


class IDCT1x(nn.Module):
    def __init__(self, dim=-1):
        super(IDCT1x, self).__init__()
        self.dctMat = None
        self.dim = dim
    def check_dct_matrix(self, d):
        if self.dctMat is None or d != self.dctMat.shape[-1]:
            self.dctMat = get_dctMatrix(d, d)
    def forward(self, x):
        if self.dim != -1 or self.dim != len(x.shape) - 1:
            x = x.transpose(self.dim, -1)
        self.check_dct_matrix(x.shape[-1])

        self.dctMat = self.dctMat.to(x.device)
        # print(x.shape, self.dctMat.shape)
        x = idct1d(x, self.dctMat)
        if self.dim != -1 or self.dim != len(x.shape) - 1:
            x = x.transpose(self.dim, -1)
        return x.contiguous()

class DCT2(nn.Module):
    def __init__(self, window_size=8, norm='ortho'):
        super(DCT2, self).__init__()
        self.dctMat = get_dctMatrix(window_size, window_size)
        self.norm = norm
        self.window_size = window_size
    def forward(self, x):
        dctMat = self.dctMat.to(x.device)
        # print(x.shape, self.dctMat.shape)
        x = dct2d(x, dctMat)
        return x
class DCT2_torch(nn.Module):
    def __init__(self, norm='ortho'):
        super(DCT2_torch, self).__init__()
        self.norm = norm
    def forward(self, x):
        # print(x.shape)
        x = dct_2d_torch(x, norm=self.norm)
        return x

class IDCT2_torch(nn.Module):
    def __init__(self, norm='ortho'):
        super(IDCT2_torch, self).__init__()
        self.norm = norm
    def forward(self, x):
        # print(x.shape)
        x = idct_2d_torch(x, norm=self.norm)
        return x

class IDCT2(nn.Module):
    def __init__(self, window_size=8, norm='ortho'):
        super(IDCT2, self).__init__()
        self.dctMat = get_dctMatrix(window_size, window_size)
        self.norm = norm
        self.window_size = window_size
    def forward(self, x):
        dctMat = self.dctMat.to(x.device)
        x = idct2d(x, dctMat)
        return x
def get_dct_init(N):
    k = - torch.arange(N, dtype=torch.float)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    return [W_r, W_i]
class DCT2x_torch(nn.Module):
    def __init__(self, norm='ortho'):
        super(DCT2x_torch, self).__init__()
        self.dctMatH = None
        self.dctMatW = None
        self.norm = norm
    def check_dct_matrix(self, h, w):
        if self.dctMatH is None or self.dctMatW is None:
            self.dctMatH = get_dct_init(h)
            self.dctMatW = get_dct_init(w)
        elif h != self.dctMatH[0].shape[-1] and w != self.dctMatW[0].shape[-1]:
            self.dctMatH = get_dct_init(h)
            self.dctMatW = get_dct_init(w)
        elif h != self.dctMatH[0].shape[-1]:
            self.dctMatH = get_dct_init(h)
            # self.dctMatH = self.dctMatH.to(x.device)
        elif w != self.dctMatW[0].shape[-1]:
            self.dctMatW = get_dct_init(w)
        # print(self.dctMatW[0].shape)
    def forward(self, x):
        h, w = x.shape[-2:]
        self.check_dct_matrix(h, w)
        # print(x.shape, self.dctMatH.shape, self.dctMatW.shape)
        x = dct_2d_torch(x, self.dctMatH, self.dctMatW, norm=self.norm)

        return x


class IDCT2x_torch(nn.Module):
    def __init__(self, norm='ortho'):
        super(IDCT2x_torch, self).__init__()
        self.dctMatH = None
        self.dctMatW = None
        self.norm = norm

    def check_dct_matrix(self, h, w):
        if self.dctMatH is None or self.dctMatW is None:
            self.dctMatH = get_dct_init(h)
            self.dctMatW = get_dct_init(w)
        elif h != self.dctMatH[0].shape[-1] and w != self.dctMatW[0].shape[-1]:
            self.dctMatH = get_dct_init(h)
            self.dctMatW = get_dct_init(w)
        elif h != self.dctMatH[0].shape[-1]:
            self.dctMatH = get_dct_init(h)
            # self.dctMatH = self.dctMatH.to(x.device)
        elif w != self.dctMatW[0].shape[-1]:
            self.dctMatW = get_dct_init(w)

    def forward(self, x):
        h, w = x.shape[-2:]
        self.check_dct_matrix(h, w)
        # print(x.shape, self.dctMatH.shape, self.dctMatW.shape)
        x = idct_2d_torch(x, self.dctMatH, self.dctMatW, norm=self.norm)

        return x


class DCT2x(nn.Module):
    def __init__(self, norm='ortho'):
        super(DCT2x, self).__init__()
        self.dctMatH = None
        self.dctMatW = None
        self.norm = norm

    def check_dct_matrix(self, h, w):
        if self.dctMatH is None or self.dctMatW is None:
            self.dctMatH = get_dctMatrix(h, h)
            self.dctMatW = get_dctMatrix(w, w)
        elif h != self.dctMatH.shape[-1] and w != self.dctMatW.shape[-1]:
            self.dctMatH = get_dctMatrix(h, h)
            self.dctMatW = get_dctMatrix(w, w)
        elif h != self.dctMatH.shape[-1]:
            self.dctMatH = get_dctMatrix(h, h)
            # self.dctMatH = self.dctMatH.to(x.device)
        elif w != self.dctMatW.shape[-1]:
            self.dctMatW = get_dctMatrix(w, w)

    def forward(self, x):
        h, w = x.shape[-2:]
        self.check_dct_matrix(h, w)
        dctMatH = self.dctMatH.to(x.device)
        dctMatW = self.dctMatW.to(x.device)
        # print(x.shape, self.dctMatH.shape, self.dctMatW.shape)
        x = dct2dx(x, dctMatW, dctMatH)

        return x


class IDCT2x(nn.Module):
    def __init__(self, norm='ortho'):
        super(IDCT2x, self).__init__()
        self.dctMatH = None
        self.dctMatW = None
        self.norm = norm

    def check_dct_matrix(self, h, w):
        if self.dctMatH is None or self.dctMatW is None:
            self.dctMatH = get_dctMatrix(h, h)
            self.dctMatW = get_dctMatrix(w, w)
        elif h != self.dctMatH.shape[-1] and w != self.dctMatW.shape[-1]:
            self.dctMatH = get_dctMatrix(h, h)
            self.dctMatW = get_dctMatrix(w, w)
        elif h != self.dctMatH.shape[-1]:
            self.dctMatH = get_dctMatrix(h, h)
            # self.dctMatH = self.dctMatH.to(x.device)
        elif w != self.dctMatW.shape[-1]:
            self.dctMatW = get_dctMatrix(w, w)

    def forward(self, x):
        h, w = x.shape[-2:]
        self.check_dct_matrix(h, w)
        dctMatH = self.dctMatH.to(x.device)
        dctMatW = self.dctMatW.to(x.device)
        x = idct2dx(x, dctMatW, dctMatH)

        return x

def d4_to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def d3_to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
def d5_to_3d(x):
    # x = x.permute(0, 2, 1, 3, 4)
    return rearrange(x, 'b c s h w -> b (s h w) c')


def d3_to_5d(x, s, h, w):
    x = rearrange(x, 'b (s h w) c -> b c s h w', s=s, h=h, w=w)
    # x = x.permute(0, 2, 1, 3, 4)
    return x
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape, bias, mu_sigma=False):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.mu_sigma = mu_sigma
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.norm_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):

        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        if self.norm_bias:
            x = (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
        else:
            x = (x - mu) / torch.sqrt(sigma+1e-5) * self.weight
        if self.mu_sigma:
            return x, mu, sigma
        else:
            return x
class WithBias_LayerNorm2x(nn.Module):
    def __init__(self, normalized_shape, bias, mu_sigma=False):
        super(WithBias_LayerNorm2x, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.mu_sigma = mu_sigma
        assert len(normalized_shape) == 1

        self.weight1 = nn.Parameter(torch.ones(normalized_shape))
        self.weight2 = nn.Parameter(torch.ones(normalized_shape))
        self.norm_bias = bias
        if bias:
            self.bias1 = nn.Parameter(torch.zeros(normalized_shape))
            self.bias2 = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):

        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        if self.norm_bias:
            x = (x - mu) / torch.sqrt(sigma+1e-5)
        else:
            x = (x - mu) / torch.sqrt(sigma+1e-5)
        x1 = x * self.weight1 + self.bias1
        x2 = x * self.weight2 + self.bias2
        return torch.cat([x1, x2], dim=-1)
class LayerNorm(nn.Module):
    def __init__(self, dim, bias=True, mu_sigma=False, out_dir=None):
        super(LayerNorm, self).__init__()
        self.mu_sigma = mu_sigma
        self.body = WithBias_LayerNorm(dim, bias, mu_sigma)
        self.out_dir = out_dir
        # if self.out_dir:
        #     self.dct = DCT2x()
    def forward(self, x):
        # if self.out_dir:
        #     save_feature(self.out_dir+'/beforeLN', x, min_max='log')
        h, w = x.shape[-2:]
        x = d4_to_3d(x)

        if self.mu_sigma:
            x, mu, sigma = self.body(x)
            return d3_to_4d(x, h, w), d3_to_4d(mu, h, w), d3_to_4d(sigma, h, w)
        else:
            x = self.body(x)
            # if self.out_dir:
            #     x = d3_to_4d(x, h, w)
            #     save_feature(self.out_dir + '/afterLN', x, min_max='log')
            #     return x
            # else:
            return d3_to_4d(x, h, w)


class LayerNorm2x(nn.Module):
    def __init__(self, dim, bias=True, mu_sigma=False, out_dir=None):
        super(LayerNorm2x, self).__init__()
        self.mu_sigma = mu_sigma
        self.body = WithBias_LayerNorm2x(dim, bias, mu_sigma)
        self.out_dir = out_dir
        # if self.out_dir:
        #     self.dct = DCT2x()

    def forward(self, x):
        # if self.out_dir:
        #     save_feature(self.out_dir+'/beforeLN', x, min_max='log')
        h, w = x.shape[-2:]
        x = d4_to_3d(x)
        x = self.body(x)
        return d3_to_4d(x, h, w)

class nnLayerNorm2d(nn.Module):
    def __init__(self, dim):
        super(nnLayerNorm2d, self).__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1).contiguous()
        x = self.norm(x).permute(0, 2, 1).contiguous()
        return x.view(b, c, h, w)

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class DownShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DownShuffle, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(in_channels, out_channels//4, kernel_size=kernel_size,
                                            stride=1, padding=(kernel_size-1)//2, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class UpShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.up = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * 4, kernel_size, padding=(kernel_size-1)//2, bias=False),
                    nn.PixelShuffle(2)
                )

    def forward(self, x):
        return self.up(x)

class Up_ConvTranspose2d(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x1):
        return self.up(x1)


class UpShuffle_freq(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels // 2
        self.up = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * 4, kernel_size, padding=(kernel_size-1)//2, bias=False),
                    nn.PixelShuffle(2)
                )
        self.freq_up = freup_Periodicpadding(in_channels, out_channels)
    def forward(self, x):
        return self.up(x) + self.freq_up(x)
class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 2, 2)
                )
    def forward(self, x):
        return self.down(x)




class freup_Areadinterpolation(nn.Module):
    def __init__(self, channels):
        super(freup_Areadinterpolation, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))

        self.post = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        N, C, H, W = x.shape

        fft_x = torch.fft.fft2(x)
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)

        amp_fuse = Mag.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        pha_fuse = Pha.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)

        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)

        output = torch.fft.ifft2(out)
        output = torch.abs(output)

        crop = torch.zeros_like(x)
        crop[:, :, 0:int(H / 2), 0:int(W / 2)] = output[:, :, 0:int(H / 2), 0:int(W / 2)]
        crop[:, :, int(H / 2):H, 0:int(W / 2)] = output[:, :, int(H * 1.5):2 * H, 0:int(W / 2)]
        crop[:, :, 0:int(H / 2), int(W / 2):W] = output[:, :, 0:int(H / 2), int(W * 1.5):2 * W]
        crop[:, :, int(H / 2):H, int(W / 2):W] = output[:, :, int(H * 1.5):2 * H, int(W * 1.5):2 * W]
        crop = F.interpolate(crop, (2 * H, 2 * W))

        return self.post(crop)


class freup_Periodicpadding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(freup_Periodicpadding, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(in_channels, in_channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(in_channels, in_channels, 1, 1, 0))

        self.post = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        # N, C, H, W = x.shape

        fft_x = torch.fft.fft2(x)
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)

        amp_fuse = torch.tile(Mag, (2, 2))
        pha_fuse = torch.tile(Pha, (2, 2))

        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)

        output = torch.fft.ifft2(out)
        output = torch.abs(output)

        return self.post(output)


class freup_Cornerdinterpolation(nn.Module):
    def __init__(self, channels):
        super(freup_Cornerdinterpolation, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))

        # self.post = nn.Conv2d(channels,channels,1,1,0)

    def forward(self, x):
        N, C, H, W = x.shape

        fft_x = torch.fft.fft2(x)  # n c h w
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)

        r = x.size(2)  # h
        c = x.size(3)  # w

        I_Mup = torch.zeros((N, C, 2 * H, 2 * W)).cuda()
        I_Pup = torch.zeros((N, C, 2 * H, 2 * W)).cuda()

        if r % 2 == 1:  # odd
            ir1, ir2 = r // 2 + 1, r // 2 + 1
        else:  # even
            ir1, ir2 = r // 2 + 1, r // 2
        if c % 2 == 1:  # odd
            ic1, ic2 = c // 2 + 1, c // 2 + 1
        else:  # even
            ic1, ic2 = c // 2 + 1, c // 2

        I_Mup[:, :, :ir1, :ic1] = Mag[:, :, :ir1, :ic1]
        I_Mup[:, :, :ir1, ic2 + c:] = Mag[:, :, :ir1, ic2:]
        I_Mup[:, :, ir2 + r:, :ic1] = Mag[:, :, ir2:, :ic1]
        I_Mup[:, :, ir2 + r:, ic2 + c:] = Mag[:, :, ir2:, ic2:]

        if r % 2 == 0:  # even
            I_Mup[:, :, ir2, :] = I_Mup[:, :, ir2, :] * 0.5
            I_Mup[:, :, ir2 + r, :] = I_Mup[:, :, ir2 + r, :] * 0.5
        if c % 2 == 0:  # even
            I_Mup[:, :, :, ic2] = I_Mup[:, :, :, ic2] * 0.5
            I_Mup[:, :, :, ic2 + c] = I_Mup[:, :, :, ic2 + c] * 0.5

        I_Pup[:, :, :ir1, :ic1] = Pha[:, :, :ir1, :ic1]
        I_Pup[:, :, :ir1, ic2 + c:] = Pha[:, :, :ir1, ic2:]
        I_Pup[:, :, ir2 + r:, :ic1] = Pha[:, :, ir2:, :ic1]
        I_Pup[:, :, ir2 + r:, ic2 + c:] = Pha[:, :, ir2:, ic2:]

        if r % 2 == 0:  # even
            I_Pup[:, :, ir2, :] = I_Pup[:, :, ir2, :] * 0.5
            I_Pup[:, :, ir2 + r, :] = I_Pup[:, :, ir2 + r, :] * 0.5
        if c % 2 == 0:  # even
            I_Pup[:, :, :, ic2] = I_Pup[:, :, :, ic2] * 0.5
            I_Pup[:, :, :, ic2 + c] = I_Pup[:, :, :, ic2 + c] * 0.5

        real = I_Mup * torch.cos(I_Pup)
        imag = I_Mup * torch.sin(I_Pup)
        out = torch.complex(real, imag)

        output = torch.fft.ifft2(out)
        output = torch.abs(output)

        return output

def window_partitions(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    if isinstance(window_size, int):
        window_size = [window_size, window_size]
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size[0], window_size[1])
    return windows


def window_reverses(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    # B = int(windows.shape[0] / (H * W / window_size / window_size))
    # print('B: ', B)
    # print(H // window_size)
    # print(W // window_size)
    if isinstance(window_size, int):
        window_size = [window_size, window_size]
    C = windows.shape[1]
    # print('C: ', C)
    x = windows.view(-1, H // window_size[0], W // window_size[1], C, window_size[0], window_size[1])
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
    return x

def window_partitionx(x, window_size):
    _, _, H, W = x.shape
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_partitions(x[:, :, :h, :w], window_size)
    b_main = x_main.shape[0]
    if h == H and w == W:
        return x_main, [b_main]
    if h != H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_r
        x_dd = x[:, :, -window_size:, -window_size:]
        b_dd = x_dd.shape[0] + b_d
        # batch_list = [b_main, b_r, b_d, b_dd]
        return torch.cat([x_main, x_r, x_d, x_dd], dim=0), [b_main, b_r, b_d, b_dd]
    if h == H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        return torch.cat([x_main, x_r], dim=0), [b_main, b_r]
    if h != H and w == W:
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_main
        return torch.cat([x_main, x_d], dim=0), [b_main, b_d]
def window_reversex(windows, window_size, H, W, batch_list):
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    # print(windows[:batch_list[0], ...].shape)
    x_main = window_reverses(windows[:batch_list[0], ...], window_size, h, w)
    B, C, _, _ = x_main.shape
    # print('windows: ', windows.shape)
    # print('batch_list: ', batch_list)
    if torch.is_complex(windows):
        res = torch.complex(torch.zeros([B, C, H, W]), torch.zeros([B, C, H, W]))
        res = res.to(windows.device)
    else:
        res = torch.zeros([B, C, H, W], dtype=windows.dtype, device=windows.device)

    res[:, :, :h, :w] = x_main
    if h == H and w == W:
        return res
    if h != H and w != W and len(batch_list) == 4:
        x_dd = window_reverses(windows[batch_list[2]:, ...], window_size, window_size, window_size)
        res[:, :, h:, w:] = x_dd[:, :, h - H:, w - W:]
        x_r = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
        res[:, :, :h, w:] = x_r[:, :, :, w - W:]
        x_d = window_reverses(windows[batch_list[1]:batch_list[2], ...], window_size, window_size, w)
        res[:, :, h:, :w] = x_d[:, :, h - H:, :]
        return res
    if w != W and len(batch_list) == 2:
        x_r = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
        res[:, :, :h, w:] = x_r[:, :, :, w - W:]
    if h != H and len(batch_list) == 2:
        x_d = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, window_size, w)
        res[:, :, h:, :w] = x_d[:, :, h - H:, :]
    return res

def window_partitionxy(x, window_size, start=[0, 0]):
    s_h, s_w = start
    assert 0 <= s_h < window_size and 0 <= s_w < window_size
    _, _, H, W = x.shape
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main, b_main = window_partitionx(x[:, :, s_h:, s_w:], window_size)
    # print(x_main.shape, b_main, x[:, :, s_h:, s_w:].shape)
    if s_h == 0 and s_w == 0:
        return x_main, b_main
    if s_h != 0 and s_w != 0:
        x_l = window_partitions(x[:, :, -h:, :window_size], window_size)
        b_l = x_l.shape[0] + b_main[-1]
        b_main.append(b_l)
        x_u = window_partitions(x[:, :, :window_size, -w:], window_size)
        b_u = x_u.shape[0] + b_l
        b_main.append(b_u)
        x_uu = x[:, :, :window_size, :window_size]
        b_uu = x_uu.shape[0] + b_u
        b_main.append(b_uu)
        # batch_list = [b_main, b_r, b_d, b_dd]
        return torch.cat([x_main, x_l, x_u, x_uu], dim=0), b_main

def window_reversexy(windows, window_size, H, W, batch_list, start=[0, 0]):
    s_h, s_w = start
    assert 0 <= s_h < window_size and 0 <= s_w < window_size

    if s_h == 0 and s_w == 0:
        x_main = window_reversex(windows, window_size, H, W, batch_list)
        return x_main
    else:
        h, w = window_size * (H // window_size), window_size * (W // window_size)
        # print(windows[:batch_list[-4], ...].shape, batch_list[:-3], H-s_h, W-s_w)
        x_main = window_reversex(windows[:batch_list[-4], ...], window_size, H-s_h, W-s_w, batch_list[:-3])
        B, C, _, _ = x_main.shape
        res = torch.zeros([B, C, H, W], device=windows.device)
        x_uu = window_reverses(windows[batch_list[-2]:, ...], window_size, window_size, window_size)
        res[:, :, :window_size, :window_size] = x_uu[:, :, :, :]
        x_l = window_reverses(windows[batch_list[-4]:batch_list[-3], ...], window_size, h, window_size)
        res[:, :, -h:, :window_size] = x_l
        x_u = window_reverses(windows[batch_list[-3]:batch_list[-2], ...], window_size, window_size, w)
        res[:, :, :window_size, -w:] = x_u[:, :, :, :]

        res[:, :, s_h:, s_w:] = x_main
        return res
class WindowPartition(nn.Module):
    def __init__(self, window_size=8, shift_size=0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
    def forward(self, x):
        if self.window_size is None:
            return x, []
        H, W = x.shape[-2:]
        if H > self.window_size and W > self.window_size:
            if not self.shift_size:
                x, batch_list = window_partitionx(x, self.window_size)
                return x, batch_list
            else:
                x, batch_list = window_partitionxy(x, self.window_size, [self.shift_size, self.shift_size])
                return x, batch_list
        else:
            return x, []

class WindowReverse(nn.Module):
    def __init__(self, window_size=8, shift_size=0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size

    def forward(self, x, H, W, batch_list):
        # print(x.shape, batch_list)
        if self.window_size is None:
            return x
        if len(batch_list) > 0 and (H > self.window_size and W > self.window_size):
            if not self.shift_size:
                x = window_reversex(x, self.window_size, H, W, batch_list)
            else:
                x = window_reversexy(x, self.window_size, H, W, batch_list, [self.shift_size, self.shift_size])
        return x

