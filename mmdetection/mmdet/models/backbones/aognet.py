""" RESEARCH ONLY LICENSE
Copyright (c) 2018-2019 North Carolina State University.
All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided
that the following conditions are met:
1. Redistributions and use are permitted for internal research purposes only, and commercial use
is strictly prohibited under this license. Inquiries regarding commercial use should be directed to the
Office of Research Commercialization at North Carolina State University, 919-215-7199,
https://research.ncsu.edu/commercialization/contact/, commercialization@ncsu.edu .
2. Commercial use means the sale, lease, export, transfer, conveyance or other distribution to a
third party for financial gain, income generation or other commercial purposes of any kind, whether
direct or indirect. Commercial use also means providing a service to a third party for financial gain,
income generation or other commercial purposes of any kind, whether direct or indirect.
3. Redistributions of source code must retain the above copyright notice, this list of conditions and
the following disclaimer.
4. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the
distribution.
5. The names “North Carolina State University”, “NCSU” and any trade-name, personal name,
trademark, trade device, service mark, symbol, image, icon, or any abbreviation, contraction or
simulation thereof owned by North Carolina State University must not be used to endorse or promote
products derived from this software without prior written permission. For written permission, please
contact trademarks@ncsu.edu.
Disclaimer: THIS SOFTWARE IS PROVIDED “AS IS” AND ANY EXPRESSED OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NORTH CAROLINA STATE UNIVERSITY BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
# The system (AOGNet) is protected via patent (pending)
# Written by Tianfu Wu
# Contact: tianfu_wu@ncsu.edu

import numpy as np

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init,
                      AttnBatchNorm2d, AttnGroupNorm)
from mmcv.runner import load_checkpoint

from mmdet.utils import get_root_logger
from mmdet.ivmcl import build_aog, build_stem_layer, AOGBlock
from ..builder import BACKBONES

from .resnext_an import Bottleneck as _Bottleneck


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Transition(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=2,
                 padding=0,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(Transition, self).__init__()

        if stride > 1:
            self.transition = nn.Sequential(
                ConvModule(in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=(kernel_size-1)//2,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg),
                nn.AvgPool2d(kernel_size=(stride, stride), stride=stride)
            )
        else:
            self.transition = Identity()

    def forward(self, x, choice=None):
        return self.transition(x)


class Bottleneck(_Bottleneck):

    def __init__(self,
                 inplanes,
                 planes,
                 outplanes,
                 drop_rate,
                 **kwargs):
        """Bottleneck block for AOGNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(inplanes, planes, **kwargs)

        self.outplanes = outplanes
        self.drop_rate = drop_rate

        _, norm3 = build_norm_layer(
            self.norm_cfg3, self.outplanes, postfix=3)

        self.conv3 = build_conv_layer(
            self.conv_cfg,
            self.width,
            self.outplanes,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        if self.drop_rate:
            self.drop = nn.Dropout2d(p=self.drop_rate, inplace=True)

    def forward(self, x, identity=None):

        def _inner_forward(x, identity=None):
            if identity is None:
                identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            if self.with_plugins:
                out = self.forward_plugin(out, self.after_bn1_plugin_names)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            if self.with_plugins:
                out = self.forward_plugin(out, self.after_bn2_plugin_names)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.drop_rate:
                out = self.drop(out)

            if self.downsample is not None:
                identity = self.downsample(identity)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, *[x, identity])
        else:
            out = _inner_forward(x, identity)

        out = self.relu(out)

        return out


@BACKBONES.register_module()
class AOGNet(nn.Module):
    """ AOGNet
    http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_AOGNets_Compositional_Grammatical_Architectures_for_Deep_Learning_CVPR_2019_paper.pdf
    """
    def __init__(self,
                 aog_cfg,
                 in_channels=3,
                 stem_type="DeepStem",
                 block='AOGBlock',
                 block_num=(2, 2, 2, 1),
                 filter_list=(32, 128, 256, 512, 824),
                 ops_t_node=('Bottleneck', 'Bottleneck',
                             'Bottleneck', 'Bottleneck'),
                 ops_and_node=('Bottleneck', 'Bottleneck',
                               'Bottleneck', 'Bottleneck'),
                 ops_or_node=('Bottleneck', 'Bottleneck',
                              'Bottleneck', 'Bottleneck'),
                 bn_ratios=(0.25, 0.25, 0.25, 0.25),
                 t_node_no_slice=(False, False, False, False),
                 t_node_handle_dblcnt=(False, False, False, False),
                 non_t_node_handle_dblcnt=(False, False, False, False),
                 or_node_reduction='sum',
                 drop_rates=(0., 0., 0.1, 0.1),
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 with_group_conv=(False, False, False, False),
                 base_width=(4, 4, 4, 4),
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 style='pytorch',
                 conv_cfg=None,
                 norm_cfg_stem=dict(type='BN', requires_grad=True),
                 norm_cfg_transition=dict(type='BN', requires_grad=True),
                 norm_cfg_ds=dict(type='BN', requires_grad=True),
                 norm_cfg1=dict(type='BN', requires_grad=True),
                 norm_cfg2=dict(type='BN', requires_grad=True),
                 norm_cfg3=dict(type='BN', requires_grad=True),
                 norm_cfg_extra=dict(type='BN', requires_grad=True),
                 num_affine_trans=(10, 10, 20, 20),
                 norm_eval=True,
                 act_cfg=dict(type='ReLU', inplace=True),
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=False,
                 use_extra_norm_ac_for_block=True,
                 handle_dbl_cnt_in_weight_init=False,
                 num_classes=0):
        """
        Separate norm_cfgs for flexibilty and ablation studies of AN
        norm_cfg_stem: for stem
        norm_cfg_transition: for transition module between aog blocks
        norm_cfg_ds: for downsampling module in a block
        norm_cfg1: for the 1st conv in a block(Basic Or Bottleneck)
        norm_cfg2: for the 2nd conv in a block(Basic Or Bottleneck)
        norm_cfg3: for the 3rd conv in a block(Bottleneck)
        norm_cfg_extra: for the extra_norm_ac module
        """
        super(AOGNet, self).__init__()
        self.num_stages = len(filter_list) - 1
        assert self.num_stages == len(aog_cfg['dims'])
        self.aogs = build_aog(aog_cfg)
        self.in_channels = in_channels
        self.stem_type = stem_type
        self.block = eval(block)
        self.block_num = block_num
        self.filter_list = filter_list
        self.ops_t_node = ops_t_node
        self.ops_and_node = ops_and_node
        self.ops_or_node = ops_or_node
        self.bn_ratios = bn_ratios
        self.t_node_no_slice = t_node_no_slice
        self.t_node_handle_dblcnt = t_node_handle_dblcnt
        self.non_t_node_handle_dblcnt = non_t_node_handle_dblcnt
        self.or_node_reduction = or_node_reduction
        self.drop_rates = drop_rates
        self.strides = strides
        self.dilations = dilations
        self.with_group_conv = with_group_conv
        self.base_width = base_width
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.style = style
        self.conv_cfg = conv_cfg
        self.norm_cfg_stem = norm_cfg_stem
        self.norm_cfg_transition = norm_cfg_transition
        self.norm_cfg_ds = norm_cfg_ds
        self.norm_cfg1 = norm_cfg1
        self.norm_cfg2 = norm_cfg2
        self.norm_cfg3 = norm_cfg3
        self.norm_cfg_extra = norm_cfg_extra
        self.num_affine_trans = num_affine_trans
        self.norm_eval = norm_eval
        self.act_cfg = act_cfg
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == self.num_stages
        self.with_cp = with_cp
        self.plugins = plugins
        self.zero_init_residual = zero_init_residual
        self.use_extra_norm_ac_for_block = use_extra_norm_ac_for_block
        self.handle_dbl_cnt_in_weight_init = handle_dbl_cnt_in_weight_init
        self.num_classes = num_classes

        self._make_stem_layer(filter_list[0])

        self._make_stages()

        self.with_classification = num_classes > 0
        if self.with_classification:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.drop = None
            if len(self.drop_rates) > self.num_stages:
                self.drop = nn.Dropout(p=self.drop_rates[-1], inplace=True)
            self.fc = nn.Linear(filter_list[-1], num_classes)

        self._freeze_stages()

        self.feat_dim = self.filter_list[-1]

        ## initialize
        self.init_weights()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (AttnBatchNorm2d, AttnGroupNorm)):
                    nn.init.normal_(m.weight_, 1., 0.1)
                    nn.init.normal_(m.bias_, 0., 0.1)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m.conv2, 'conv_offset'):
                        constant_init(m.conv2.conv_offset, 0)

            if self.handle_dbl_cnt_in_weight_init:
                import re
                for name_, m in self.named_modules():
                    if 'node' in name_:
                        idx = re.findall(r'\d+', name_)
                        stage_id = int(idx[-3])
                        node_id = int(idx[-1])
                        npaths = self.aogs[stage_id-1].node_set[node_id].npaths
                        if npaths > 1:
                            scale = 1.0 / npaths
                            with torch.no_grad():
                                for ch in m.modules():
                                    if isinstance(ch, nn.Conv2d):
                                        ch.weight.mul_(scale)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, (Bottleneck, BottleneckV2)):
                        if isinstance(m.norm3,
                                        (AttnBatchNorm2d, AttnGroupNorm)):
                            nn.init.constant_(m.norm3.weight_, 0.)
                            nn.init.constant_(m.norm3.bias_, 0.)
                        else:
                            constant_init(m.norm3, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def _make_stem_layer(self, planes):
        if 'Attn' in self.norm_cfg_stem['type']:
            self.norm_cfg_stem['num_affine_trans'] = self.num_affine_trans[0]
        stem_cfg = dict(type=self.stem_type,
                        inplanes=self.in_channels,
                        planes=planes,
                        norm_cfg=self.norm_cfg_stem)
        self.stem = build_stem_layer(stem_cfg)

    def _make_stages(self):
        self.aog_layers = []
        for stage_id in range(self.num_stages):
            aog = self.aogs[stage_id]
            dim = aog.param.grid_wd
            in_channels = self.filter_list[stage_id]
            out_channels = self.filter_list[stage_id + 1]

            assert in_channels % dim == 0 and out_channels % dim == 0
            step_channels = (
                out_channels - in_channels) // self.block_num[stage_id]
            if step_channels % dim != 0:
                low = (step_channels // dim) * dim
                high = (step_channels // dim + 1) * dim
                if (step_channels-low) <= (high-step_channels):
                    step_channels = low
                else:
                    step_channels = high

            if 'Attn' in self.norm_cfg_transition['type']:
                self.norm_cfg_transition['num_affine_trans'] = \
                    self.num_affine_trans[stage_id]
            if 'Attn' in self.norm_cfg_ds['type']:
                self.norm_cfg_ds['num_affine_trans'] = \
                    self.num_affine_trans[stage_id]
            if 'Attn' in self.norm_cfg1['type']:
                self.norm_cfg1['num_affine_trans'] = \
                    self.num_affine_trans[stage_id]
            if 'Attn' in self.norm_cfg2['type']:
                self.norm_cfg2['num_affine_trans'] = \
                    self.num_affine_trans[stage_id]
            if 'Attn' in self.norm_cfg3['type']:
                self.norm_cfg3['num_affine_trans'] = \
                    self.num_affine_trans[stage_id]
            if 'Attn' in self.norm_cfg_extra['type']:
                self.norm_cfg_extra['num_affine_trans'] = \
                    self.num_affine_trans[stage_id]

            aog_layer = []

            for j in range(self.block_num[stage_id]):
                stride = self.strides[stage_id] if j == 0 else 1
                outchannels = (in_channels + step_channels) if \
                    j < self.block_num[stage_id]-1 else out_channels

                # transition
                aog_layer.append(
                    Transition(in_channels,
                            in_channels,
                            kernel_size=1,
                            stride=stride,
                            padding=0,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg_transition,
                            act_cfg=self.act_cfg))

                # blocks
                bn_ratio = self.bn_ratios[stage_id]
                aog_layer.append(
                    self.block(stage_id=stage_id+1,
                               block_id=j+1,
                               aog=aog,
                               op_t_node=eval(self.ops_t_node[stage_id]),
                               op_and_node=eval(self.ops_and_node[stage_id]),
                               op_or_node=eval(self.ops_or_node[stage_id]),
                               inplanes=in_channels,
                               outplanes=outchannels,
                               bn_ratio=bn_ratio,
                               t_node_no_slice=self.t_node_no_slice[stage_id],
                               t_node_handle_dblcnt=\
                                   self.t_node_handle_dblcnt[stage_id],
                               non_t_node_handle_dblcnt=\
                                   self.non_t_node_handle_dblcnt[stage_id],
                               or_node_reduction=self.or_node_reduction,
                               drop_rate=self.drop_rates[stage_id],
                               stride=1,
                               dilation=self.dilations[stage_id],
                               with_group_conv=self.with_group_conv[stage_id],
                               base_width=self.base_width[stage_id],
                               style=self.style,
                               with_cp=self.with_cp,
                               conv_cfg=self.conv_cfg,
                               norm_cfg_ds=self.norm_cfg_ds,
                               norm_cfg1=self.norm_cfg1,
                               norm_cfg2=self.norm_cfg2,
                               norm_cfg3=self.norm_cfg3,
                               norm_cfg_extra=self.norm_cfg_extra,
                               act_cfg=self.act_cfg,
                               dcn=self.dcn,
                               plugins=self.plugins,
                               use_extra_norm_ac_for_block=\
                                   self.use_extra_norm_ac_for_block))

                in_channels = outchannels

            layer_name = f'layer{stage_id + 1}'
            self.add_module(layer_name, nn.Sequential(*aog_layer))
            self.aog_layers.append(layer_name)

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

    def forward(self, x):
        y = self.stem(x)

        outs = []
        for i, layer_name in enumerate(self.aog_layers):
            aog_layer = getattr(self, layer_name)
            y = aog_layer(y)
            if i in self.out_indices:
                outs.append(y)

        if self.with_classification:
            y = self.avgpool(y)
            y = y.reshape(y.size(0), -1)
            if self.drop is not None:
                y = self.drop(y)
            y = self.fc(y)
            return y

        return tuple(outs)

    def train(self, mode=True):
        super(AOGNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
