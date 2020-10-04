""" modified from torchvision (3e69462)
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init, AttnBatchNorm2d, AttnGroupNorm)
from mmcv.runner import load_checkpoint

from mmdet.utils import get_root_logger
from mmdet.ivmcl import build_stem_layer
from ..builder import BACKBONES


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes,
                 kernel_size=3, stride=1, groups=1,
                 norm_cfg=dict(type='BN')):
        padding = (kernel_size - 1) // 2

        _, norm = build_norm_layer(
            norm_cfg, out_planes)

        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                      padding, groups=groups, bias=False),
            norm,
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio,
                 norm_cfg1=dict(type='BN'),
                 norm_cfg2=dict(type='BN'),
                 norm_cfg3=dict(type='BN')):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        _, norm3 = build_norm_layer(norm_cfg3, oup)

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim,
                                     kernel_size=1, norm_cfg=norm_cfg1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride,
                       groups=hidden_dim, norm_cfg=norm_cfg2),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm3,
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


@BACKBONES.register_module()
class MobileNetV2(nn.Module):
    def __init__(self,
                 in_channels=3,
                 num_classes=0,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_cfg_stem=dict(type='BN', requires_grad=True),
                 norm_cfg1=dict(type='BN', requires_grad=True),
                 norm_cfg2=dict(type='BN', requires_grad=True),
                 norm_cfg3=dict(type='BN', requires_grad=True),
                 norm_cfg_final=dict(type='BN', requires_grad=True),
                 num_affine_trans=(5, 5, 5, 5, 10, 10, 10),
                 norm_eval=True,
                 frozen_stages=-1,
                 out_indices=(1, 2, 4, 6)):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        self.in_channels = in_channels
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages
        self.out_indices = out_indices
        assert max(out_indices) < len(inverted_residual_setting)

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or \
            len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(
            input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest)
        if 'Attn' in norm_cfg_stem['type']:
            norm_cfg_stem['num_affine_trans'] = num_affine_trans[0]
        self.stem = ConvBNReLU(
            in_channels, input_channel, stride=2, norm_cfg=norm_cfg_stem)

        # building inverted residual blocks
        self.res_layers = []
        for l, (t, c, n, s) in enumerate(inverted_residual_setting):
            if 'Attn' in norm_cfg1['type']:
                norm_cfg1['num_affine_trans'] = num_affine_trans[l]
            if 'Attn' in norm_cfg2['type']:
                norm_cfg2['num_affine_trans'] = num_affine_trans[l]
            if 'Attn' in norm_cfg3['type']:
                norm_cfg3['num_affine_trans'] = num_affine_trans[l]

            output_channel = _make_divisible(c * width_mult, round_nearest)
            features = []
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel,
                                      stride, expand_ratio=t,
                                      norm_cfg1=norm_cfg1,
                                      norm_cfg2=norm_cfg2,
                                      norm_cfg3=norm_cfg3))
                input_channel = output_channel
            if l == len(inverted_residual_setting)-1:
                # building last several layers
                if 'Attn' in norm_cfg_final['type']:
                    norm_cfg_final['num_affine_trans'] = num_affine_trans[-1]
                features.append(ConvBNReLU(input_channel, self.last_channel,
                                        kernel_size=1, norm_cfg=norm_cfg_final))
            layer_name = f'layer{l+1}'
            self.add_module(layer_name, nn.Sequential(*features))
            self.res_layers.append(layer_name)

        # building classifier
        self.with_classification = num_classes > 0
        if self.with_classification:
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.last_channel, num_classes),
            )

        self._freeze_stages()

        self.init_weights()

        self.feat_dim = self.last_channel

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages+1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (AttnBatchNorm2d, AttnGroupNorm)):
                    nn.init.normal_(m.weight_, 1., 0.1)
                    nn.init.normal_(m.bias_, 0., 0.1)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # # This exists since TorchScript doesn't support inheritance, so the superclass method
        # # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        # x = self.features(x)
        # # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        # x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        # x = self.classifier(x)
        # return x

        y = self.stem(x)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            y = res_layer(y)
            if i in self.out_indices:
                outs.append(y)

        if self.with_classification:
            y = F.adaptive_avg_pool2d(y, (1, 1))
            y = torch.flatten(y, 1)
            y = self.classifier(y)
            return y

        return tuple(outs)

    def forward(self, x):
        return self._forward_impl(x)

    def train(self, mode=True):
        super(MobileNetV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, (AttnBatchNorm2d, AttnGroupNorm, _BatchNorm)):
                    m.eval()

