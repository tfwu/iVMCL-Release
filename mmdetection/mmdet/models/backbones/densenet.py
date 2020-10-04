"""  modified from torchvision (c558be6)
"""

# import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
# from .utils import load_state_dict_from_url
from torch import Tensor
from torch.jit.annotations import List
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init, AttnBatchNorm2d, AttnGroupNorm)
from mmcv.runner import load_checkpoint

from mmdet.utils import get_root_logger
from mmdet.ivmcl import build_stem_layer
from ..builder import BACKBONES


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate,
                 norm_cfg1=dict(type='BN'),
                 norm_cfg2=dict(type='BN'),
                 memory_efficient=False):
        super(_DenseLayer, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg1, num_input_features, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg2, bn_size * growth_rate, postfix=2)

        self.add_module(self.norm1_name, norm1),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module(self.norm2_name, norm2),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (Tensor) -> (Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_blocks, num_input_features, bn_size, growth_rate,
                 drop_rate,
                 norm_cfg1=dict(type='BN'),
                 norm_cfg2=dict(type='BN'),
                 memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_blocks):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                norm_cfg1=norm_cfg1,
                norm_cfg2=norm_cfg2,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features,
                 norm_cfg=dict(type='BN')):
        super(_Transition, self).__init__()
        self.norm_name, norm = build_norm_layer(norm_cfg, num_input_features)
        self.add_module(self.norm_name, norm)
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

    @property
    def norm(self):
        return getattr(self, self.norm_name)


@BACKBONES.register_module()
class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    arch_settings = {
        121: (32, (6, 12, 24, 16), 64),
        161: (48, (6, 12, 36, 24), 96),
        169: (32, (6, 12, 32, 32), 64),
        201: (32, (6, 12, 48, 32), 64)
    }

    def __init__(self, depth,
                 in_channels=3, stem_type="Stem",
                 bn_size=4, drop_rate=0, num_classes=0,
                 norm_cfg_transition=dict(type='BN', requires_grad=True),
                 norm_cfg1=dict(type='BN', requires_grad=True),
                 norm_cfg2=dict(type='BN', requires_grad=True),
                 norm_cfg_final=dict(type='BN', requires_grad=True),
                 num_affine_trans=(10, 10, 20, 20),
                 norm_eval=True,
                 frozen_stages=-1,
                 out_indices=(0, 1, 2, 3),
                 memory_efficient=False):

        super(DenseNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for densenet')

        growth_rate, block_config, num_init_features = self.arch_settings[depth]

        self.in_channels = in_channels
        self.stem_type = stem_type
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages
        self.out_indices = out_indices
        assert max(out_indices) < len(block_config)

        # First convolution
        self._make_stem_layer(num_init_features)

        # Each denseblock
        num_features = num_init_features
        self.dense_layers = []
        for i, num_blocks in enumerate(block_config):
            if 'Attn' in norm_cfg_transition['type']:
                norm_cfg_transition['num_affine_trans'] = num_affine_trans[i]
            if 'Attn' in norm_cfg1['type']:
                norm_cfg1['num_affine_trans'] = num_affine_trans[i]
            if 'Attn' in norm_cfg2['type']:
                norm_cfg2['num_affine_trans'] = num_affine_trans[i]

            block = [_DenseBlock(
                num_blocks=num_blocks,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                norm_cfg1=norm_cfg1,
                norm_cfg2=norm_cfg2,
                memory_efficient=memory_efficient
            )]

            num_features = num_features + num_blocks * growth_rate
            if i != len(block_config) - 1:
                block.append(_Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2,
                                    norm_cfg=norm_cfg_transition))
                num_features = num_features // 2
            else:
                # Final  norm
                if 'Attn' in norm_cfg_final['type']:
                    norm_cfg_final['num_affine_trans'] = num_affine_trans[-1]
                _, norm5 = build_norm_layer(
                    norm_cfg_final, num_features, postfix=5)
                block.append(norm5)
                block.append(nn.ReLU(inplace=True))

            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, nn.Sequential(*block))
            self.dense_layers.append(layer_name)

        # Linear layer
        self.with_classification = num_classes > 0
        if self.with_classification:
            self.classifier = nn.Linear(num_features, num_classes)

        self._freeze_stages()

        self.init_weights()

        self.feat_dim = num_features

    def _make_stem_layer(self, planes):
        stem_cfg = dict(type=self.stem_type,
                        inplanes=self.in_channels,
                        planes=planes)
        self.stem = build_stem_layer(stem_cfg)

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

    def forward(self, x):
        y = self.stem(x)

        outs = []
        for i, layer_name in enumerate(self.dense_layers):
            dense_layer = getattr(self, layer_name)
            y = dense_layer(y)
            if i in self.out_indices:
                outs.append(y)

        if self.with_classification:
            y = F.adaptive_avg_pool2d(y, (1, 1))
            y = torch.flatten(y, 1)
            y = self.classifier(y)
            return y

        return tuple(outs)

    def train(self, mode=True):
        super(DenseNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, (AttnBatchNorm2d, AttnGroupNorm, _BatchNorm)):
                    m.eval()


# def _load_state_dict(model, model_url, progress):
#     # '.'s are no longer allowed in module names, but previous _DenseLayer
#     # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
#     # They are also in the checkpoints in model_urls. This pattern is used
#     # to find such keys.
#     pattern = re.compile(
#         r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

#     state_dict = load_state_dict_from_url(model_url, progress=progress)
#     for key in list(state_dict.keys()):
#         res = pattern.match(key)
#         if res:
#             new_key = res.group(1) + res.group(2)
#             state_dict[new_key] = state_dict[key]
#             del state_dict[key]
#     model.load_state_dict(state_dict)



