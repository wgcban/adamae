from functools import partial
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from typing import Any, Callable, Optional, Tuple, Union

from model_configs.config_movinet import _C as movinet_cfg
from collections import OrderedDict
from torch.nn.modules.utils import _triple, _pair

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size = (self.tubelet_size,  patch_size[0],patch_size[1]), 
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 

#Movinet
class Hardsigmoid(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        x = (0.2 * x + 0.5).clamp(min=0.0, max=1.0)
        return x

class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class CausalModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.activation = None

    def reset_activation(self) -> None:
        self.activation = None


class TemporalCGAvgPool3D(CausalModule):
    def __init__(self,) -> None:
        super().__init__()
        self.n_cumulated_values = 0
        self.register_forward_hook(self._detach_activation)

    def forward(self, x: Tensor) -> Tensor:
        input_shape = x.shape
        device = x.device
        cumulative_sum = torch.cumsum(x, dim=2)
        if self.activation is None:
            self.activation = cumulative_sum[:, :, -1:].clone()
        else:
            #print('cumulative_sum.shape = {}, cumulative_sum = {}'.format(cumulative_sum.shape, self.activation.shape))
            cumulative_sum += self.activation
            self.activation = cumulative_sum[:, :, -1:].clone()
        divisor = (torch.arange(1, input_shape[2]+1,
                   device=device)[None, None, :, None, None]
                   .expand(x.shape))
        x = cumulative_sum / (self.n_cumulated_values + divisor)
        self.n_cumulated_values += input_shape[2]
        return x

    @staticmethod
    def _detach_activation(module: CausalModule,
                           input: Tensor,
                           output: Tensor) -> None:
        module.activation.detach_()

    def reset_activation(self) -> None:
        super().reset_activation()
        self.n_cumulated_values = 0


class Conv2dBNActivation(nn.Sequential):
    def __init__(
                 self,
                 in_planes: int,
                 out_planes: int,
                 *,
                 kernel_size: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None,
                 **kwargs: Any,
                 ) -> None:
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        if norm_layer is None:
            norm_layer = nn.Identity
        if activation_layer is None:
            activation_layer = nn.Identity
        self.kernel_size = kernel_size
        self.stride = stride
        dict_layers = OrderedDict({
                            "conv2d": nn.Conv2d(in_planes, out_planes,
                                                kernel_size=kernel_size,
                                                stride=stride,
                                                padding=padding,
                                                groups=groups,
                                                **kwargs),
                            "norm": norm_layer(out_planes, eps=0.001),
                            "act": activation_layer()
                            })

        self.out_channels = out_planes
        super(Conv2dBNActivation, self).__init__(dict_layers)


class Conv3DBNActivation(nn.Sequential):
    def __init__(
                 self,
                 in_planes: int,
                 out_planes: int,
                 *,
                 kernel_size: Union[int, Tuple[int, int, int]],
                 padding: Union[int, Tuple[int, int, int]],
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None,
                 **kwargs: Any,
                 ) -> None:
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        if norm_layer is None:
            norm_layer = nn.Identity
        if activation_layer is None:
            activation_layer = nn.Identity
        self.kernel_size = kernel_size
        self.stride = stride

        dict_layers = OrderedDict({
                                "conv3d": nn.Conv3d(in_planes, out_planes,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=padding,
                                                    groups=groups,
                                                    **kwargs),
                                "norm": norm_layer(out_planes, eps=0.001),
                                "act": activation_layer()
                                })

        self.out_channels = out_planes
        super(Conv3DBNActivation, self).__init__(dict_layers)


class ConvBlock3D(CausalModule):
    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            *,
            kernel_size: Union[int, Tuple[int, int, int]],
            tf_like: bool,
            causal: bool,
            conv_type: str,
            padding: Union[int, Tuple[int, int, int]] = 0,
            stride: Union[int, Tuple[int, int, int]] = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            activation_layer: Optional[Callable[..., nn.Module]] = None,
            bias: bool = False,
            **kwargs: Any,
            ) -> None:
        super().__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        self.conv_2 = None
        if tf_like:
            # We neek odd kernel to have even padding
            # and stride == 1 to precompute padding,
            if kernel_size[0] % 2 == 0:
                raise ValueError('tf_like supports only odd'
                                 + ' kernels for temporal dimension')
            padding = ((kernel_size[0]-1)//2, 0, 0)
            if stride[0] != 1:
                raise ValueError('illegal stride value, tf like supports'
                                 + ' only stride == 1 for temporal dimension')
            if stride[1] > kernel_size[1] or stride[2] > kernel_size[2]:
                # these values are not tested so should be avoided
                raise ValueError('tf_like supports only'
                                 + '  stride <= of the kernel size')

        if causal is True:
            padding = (0, padding[1], padding[2])
        if conv_type != "2plus1d" and conv_type != "3d":
            raise ValueError("only 2plus2d or 3d are "
                             + "allowed as 3d convolutions")

        if conv_type == "2plus1d":
            self.conv_1 = Conv2dBNActivation(in_planes,
                                             out_planes,
                                             kernel_size=(kernel_size[1],
                                                          kernel_size[2]),
                                             padding=(padding[1],
                                                      padding[2]),
                                             stride=(stride[1], stride[2]),
                                             activation_layer=activation_layer,
                                             norm_layer=norm_layer,
                                             bias=bias,
                                             **kwargs)
            if kernel_size[0] > 1:
                self.conv_2 = Conv2dBNActivation(in_planes,
                                                 out_planes,
                                                 kernel_size=(kernel_size[0],
                                                              1),
                                                 padding=(padding[0], 0),
                                                 stride=(stride[0], 1),
                                                 activation_layer=activation_layer,
                                                 norm_layer=norm_layer,
                                                 bias=bias,
                                                 **kwargs)
        elif conv_type == "3d":
            self.conv_1 = Conv3DBNActivation(in_planes,
                                             out_planes,
                                             kernel_size=kernel_size,
                                             padding=padding,
                                             activation_layer=activation_layer,
                                             norm_layer=norm_layer,
                                             stride=stride,
                                             bias=bias,
                                             **kwargs)
        self.padding = padding
        self.kernel_size = kernel_size
        self.dim_pad = self.kernel_size[0]-1
        self.stride = stride
        self.causal = causal
        self.conv_type = conv_type
        self.tf_like = tf_like

    def _forward(self, x, mask=None):
        device = x.device
        
        # Conv Masking
        if mask != None:
            B, C, T, H, W = x.shape
            b, c, t, h, w = mask.shape
            mask1 = repeat(mask, 'b c t h w -> b c t (h h2) (w w2)', h2=H//h, w2=W//w)
            x=x*mask1
        
        if self.dim_pad > 0 and self.conv_2 is None and self.causal is True:
            x = self._cat_stream_buffer(x, device)
        shape_with_buffer = x.shape
        
        if self.conv_type == "2plus1d":
            x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.conv_1(x)
        if self.conv_type == "2plus1d":
            x = rearrange(x,
                          "(b t) c h w -> b c t h w",
                          t=shape_with_buffer[2])

            if self.conv_2 is not None:
                # Conv masking
                if mask != None:
                    B, C, T, H, W = x.shape
                    b, c, t, h, w = mask.shape
                    mask2 = repeat(mask, 'b c t h w -> b c t (h h2) (w w2)', h2=H//h, w2=W//w)
                    x=x*mask2
                
                if self.dim_pad > 0 and self.causal is True:
                    x = self._cat_stream_buffer(x, device)
                w = x.shape[-1]
                x = rearrange(x, "b c t h w -> b c t (h w)")
                x = self.conv_2(x)
                x = rearrange(x, "b c t (h w) -> b c t h w", w=w)
        return x

    def forward(self, x=None, mask=None):
        if self.tf_like:
            x = same_padding(x, x.shape[-2], x.shape[-1],
                             self.stride[-2], self.stride[-1],
                             self.kernel_size[-2], self.kernel_size[-1])    
        x = self._forward(x, mask)
        return x

    def _cat_stream_buffer(self, x: Tensor, device: torch.device) -> Tensor:
        if self.activation is None:
            self._setup_activation(x.shape)
        x = torch.cat((self.activation.to(device), x), 2)
        self._save_in_activation(x)
        return x

    def _save_in_activation(self, x: Tensor) -> None:
        assert self.dim_pad > 0
        self.activation = x[:, :, -self.dim_pad:, ...].clone().detach()

    def _setup_activation(self, input_shape: Tuple[float, ...]) -> None:
        assert self.dim_pad > 0
        self.activation = torch.zeros(*input_shape[:2],  # type: ignore
                                      self.dim_pad,
                                      *input_shape[3:])
# TODO add requirements
# TODO create a train sample, just so that we can test the training


class SqueezeExcitation(nn.Module):

    def __init__(self, input_channels: int,  # TODO rename activations
                 activation_2: nn.Module,
                 activation_1: nn.Module,
                 conv_type: str,
                 causal: bool,
                 squeeze_factor: int = 4,
                 bias: bool = True) -> None:
        super().__init__()
        self.causal = causal
        se_multiplier = 2 if causal else 1
        squeeze_channels = _make_divisible(input_channels
                                           // squeeze_factor
                                           * se_multiplier, 8)
        self.temporal_cumualtive_GAvg3D = TemporalCGAvgPool3D()
        self.fc1 = ConvBlock3D(input_channels*se_multiplier,
                               squeeze_channels,
                               kernel_size=(1, 1, 1),
                               padding=0,
                               tf_like=False,
                               causal=causal,
                               conv_type=conv_type,
                               bias=bias)
        self.activation_1 = activation_1()
        self.activation_2 = activation_2()
        self.fc2 = ConvBlock3D(squeeze_channels,
                               input_channels,
                               kernel_size=(1, 1, 1),
                               padding=0,
                               tf_like=False,
                               causal=causal,
                               conv_type=conv_type,
                               bias=bias)

    def _scale(self, input: Tensor) -> Tensor:
        if self.causal:
            x_space = torch.mean(input, dim=[3, 4], keepdim=True)
            scale = self.temporal_cumualtive_GAvg3D(x_space)
            scale = torch.cat((scale, x_space), dim=1)
        else:
            scale = F.adaptive_avg_pool3d(input, 1)
        scale = self.fc1(scale)
        scale = self.activation_1(scale)
        scale = self.fc2(scale)
        return self.activation_2(scale)

    def forward(self, input):
        scale = self._scale(input)
        return scale * input


def _make_divisible(v: float,
                    divisor: int,
                    min_value: Optional[int] = None
                    ) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def same_padding(x: Tensor,
                 in_height: int, in_width: int,
                 stride_h: int, stride_w: int,
                 filter_height: int, filter_width: int) -> Tensor:
    if (in_height % stride_h == 0):
        pad_along_height = max(filter_height - stride_h, 0)
    else:
        pad_along_height = max(filter_height - (in_height % stride_h), 0)
    if (in_width % stride_w == 0):
        pad_along_width = max(filter_width - stride_w, 0)
    else:
        pad_along_width = max(filter_width - (in_width % stride_w), 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    padding_pad = (pad_left, pad_right, pad_top, pad_bottom)
    return torch.nn.functional.pad(x, padding_pad)


class tfAvgPool3D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.avgf = nn.AvgPool3d((1, 3, 3), stride=(1, 2, 2))

    def forward(self, x=None) -> Tensor:
        if x.shape[-1] != x.shape[-2]:
            raise RuntimeError('only same shape for h and w ' +
                               'are supported by avg with tf_like')
        if x.shape[-1] != x.shape[-2]:
            raise RuntimeError('only same shape for h and w ' +
                               'are supported by avg with tf_like')
        f1 = x.shape[-1] % 2 != 0
        if f1:
            padding_pad = (0, 0, 0, 0)
        else:
            padding_pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, padding_pad)
        if f1:
            x = torch.nn.functional.avg_pool3d(x,
                                               (1, 3, 3),
                                               stride=(1, 2, 2),
                                               count_include_pad=False,
                                               padding=(0, 1, 1))
        else:
            x = self.avgf(x)
            x[..., -1] = x[..., -1] * 9/6
            x[..., -1, :] = x[..., -1, :] * 9/6
        return x


class BasicBneck(nn.Module):
    def __init__(self,
                 cfg: "CfgNode",
                 causal: bool,
                 tf_like: bool,
                 conv_type: str,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None,
                 ) -> None:
        super().__init__()
        assert type(cfg.stride) is tuple
        if (not cfg.stride[0] == 1
                or not (1 <= cfg.stride[1] <= 2)
                or not (1 <= cfg.stride[2] <= 2)):
            raise ValueError('illegal stride value')
        self.res = None

        layers = []
        if cfg.expanded_channels != cfg.out_channels:
            # expand
            self.expand = ConvBlock3D(
                in_planes=cfg.input_channels,
                out_planes=cfg.expanded_channels,
                kernel_size=(1, 1, 1),
                padding=(0, 0, 0),
                causal=causal,
                conv_type=conv_type,
                tf_like=tf_like,
                norm_layer=norm_layer,
                activation_layer=activation_layer
                )
        # deepwise
        self.deep = ConvBlock3D(
            in_planes=cfg.expanded_channels,
            out_planes=cfg.expanded_channels,
            kernel_size=cfg.kernel_size,
            padding=cfg.padding,
            stride=cfg.stride,
            groups=cfg.expanded_channels,
            causal=causal,
            conv_type=conv_type,
            tf_like=tf_like,
            norm_layer=norm_layer,
            activation_layer=activation_layer
            )
        # SE
        self.se = SqueezeExcitation(cfg.expanded_channels,
                                    causal=causal,
                                    activation_1=activation_layer,
                                    activation_2=(nn.Sigmoid
                                                  if conv_type == "3d"
                                                  else Hardsigmoid),
                                    conv_type=conv_type
                                    )
        # project
        self.project = ConvBlock3D(
            cfg.expanded_channels,
            cfg.out_channels,
            kernel_size=(1, 1, 1),
            padding=(0, 0, 0),
            causal=causal,
            conv_type=conv_type,
            tf_like=tf_like,
            norm_layer=norm_layer,
            activation_layer=nn.Identity
            )

        if not (cfg.stride == (1, 1, 1)
                and cfg.input_channels == cfg.out_channels):
            if cfg.stride != (1, 1, 1):
                if tf_like:
                    layers.append(tfAvgPool3D())
                else:
                    layers.append(nn.AvgPool3d((1, 3, 3),
                                  stride=cfg.stride,
                                  padding=cfg.padding_avg))
            layers.append(ConvBlock3D(
                    in_planes=cfg.input_channels,
                    out_planes=cfg.out_channels,
                    kernel_size=(1, 1, 1),
                    padding=(0, 0, 0),
                    norm_layer=norm_layer,
                    activation_layer=nn.Identity,
                    causal=causal,
                    conv_type=conv_type,
                    tf_like=tf_like
                    ))
            self.res = nn.Sequential(*layers)
        # ReZero
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, input=None, mask=None):
        if self.res is not None:
            residual = self.res(input)
        else:
            residual = input
        if self.expand is not None:
            x = self.expand(input)
        else:
            x = input
        x = self.deep(x, mask)
        x = self.se(x)
        x = self.project(x, mask)
        result = residual + self.alpha * x
        return result


class MoviNet(nn.Module):
    def __init__(self,
                 cfg: "CfgNode",
                 causal: bool = True,
                 pretrained: bool = False,
                 conv_type: str = "3d",
                 tf_like: bool = False,
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=0, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer_mlp=nn.LayerNorm, init_values=0., 
                 tubelet_size=2,
                 all_frames=16,
                 use_learnable_pos_emb=False,
                 use_mean_pooling=True,
                 init_scale=0.,
                 ) -> None:
        super().__init__()
        """
        causal: causal mode
        pretrained: pretrained models
        If pretrained is True:
            num_classes is set to 600,
            conv_type is set to "3d" if causal is False,
                "2plus1d" if causal is True
            tf_like is set to True
        num_classes: number of classes for classifcation
        conv_type: type of convolution either 3d or 2plus1d
        tf_like: tf_like behaviour, basically same padding for convolutions
        """
        self.cfg = cfg
        self.depth = depth

        if pretrained:
            tf_like = True
            num_classes = 600
            conv_type = "2plus1d" if causal else "3d"
        blocks_dic = OrderedDict()

        norm_layer_movinet = nn.BatchNorm3d if conv_type == "3d" else nn.BatchNorm2d
        activation_layer = Swish if conv_type == "3d" else nn.Hardswish
        
        # Params
        self.img_size = img_size
        self.patch_size = [patch_size, patch_size]
        self.tubelet_size = tubelet_size

        # conv1
        self.conv1 = ConvBlock3D(
            in_planes=cfg.conv1.input_channels,
            out_planes=cfg.conv1.out_channels,
            kernel_size=cfg.conv1.kernel_size,
            stride=cfg.conv1.stride,
            padding=cfg.conv1.padding,
            causal=causal,
            conv_type=conv_type,
            tf_like=tf_like,
            norm_layer=norm_layer_movinet,
            activation_layer=activation_layer
            )
        
        # blocks
        blocks_dic = []
        for i, block in enumerate(cfg.blocks):
            for j, basicblock in enumerate(block):
                blocks_dic.append(BasicBneck(basicblock,
                                                      causal=causal,
                                                      conv_type=conv_type,
                                                      tf_like=tf_like,
                                                      norm_layer=norm_layer_movinet,
                                                      activation_layer=activation_layer
                                                      ))
        self.conv_blocks = nn.ModuleList(blocks_dic) #nn.Sequential(blocks_dic)
        
        # conv7
        self.conv7 = ConvBlock3D(
            in_planes=cfg.conv7.input_channels,
            out_planes=embed_dim,
            kernel_size=cfg.conv7.kernel_size,
            stride=cfg.conv7.stride,
            padding=cfg.conv7.padding,
            causal=causal,
            conv_type=conv_type,
            tf_like=tf_like,
            norm_layer=norm_layer_movinet,
            activation_layer=activation_layer
            )

        #Transformer blocks
        self.num_patches= (224//16)*(224//16)*(16//2)
        self.pos_embed  = get_sinusoid_encoding_table(self.num_patches, embed_dim)
        dpr             = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.trans_blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer_mlp,
                init_values=init_values)
            for i in range(depth)])
        
        self.norm = nn.Identity() if use_mean_pooling else norm_layer_mlp(embed_dim)
        self.fc_norm = norm_layer_mlp(embed_dim) if use_mean_pooling else None
        
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        trunc_normal_(self.head.weight, std=.02)
        self.apply(self._weight_init)

        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)
        
    def get_num_layers(self):
        return len(self.cfg.blocks)+self.depth
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def my_sequ(self, blocks_dic):
        return nn.Sequential(blocks_dic)

    def avg(self, x: Tensor) -> Tensor:
        if self.causal:
            avg = F.adaptive_avg_pool3d(x, (x.shape[2], 1, 1))
            avg = self.cgap(avg)[:, :, -1:]
        else:
            avg = F.adaptive_avg_pool3d(x, 1)
        return avg

    def _weight_init(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def _forward_impl(self, x=None):
        #Convolutional
        x = self.conv1(x) #8, 16, 16, 112, 112
        
        for block in self.conv_blocks:
            x = block(x) #8, 144, 16, 14, 14
        x = self.conv7(x) #8, 768, 8, 14, 14
        
        #Transformer bottleneck
        x       = x.reshape(x.shape[0], -1, x.shape[1]) #8, 1568 (8x14x14), 768
        x       = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        
        for trans_blk in self.trans_blocks:
            x = trans_blk(x) # 8, 1568, 768
        
        x = self.norm(x)
        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))
        else:
            return x[:, 0]

        return x

    def forward(self, x: Tensor) -> Tensor:
        #print("Input to MoviNet: {}".format(x.shape))
        #print("Mask to MoviNet: {}".format(mask.shape)) #8, 392 (7*7*8)

        x = self._forward_impl(x) #shape x: 8, 3, 16, 224, 224
        x = self.head(x)

        return x

    @staticmethod
    def _clean_activation_buffers(m):
        if issubclass(type(m), CausalModule):
            m.reset_activation()

    def clean_activation_buffers(self) -> None:
        self.apply(self._clean_activation_buffers)

@register_model
def MoviNet_A2(pretrained=False, **kwargs):
    model = MoviNet(
        img_size=224,
        patch_size=16, 
        embed_dim=768, 
        depth=1, 
        num_heads=12,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer_mlp=partial(nn.LayerNorm, eps=1e-6),
        cfg=movinet_cfg.MODEL.MoViNetA2, **kwargs)
    return model
