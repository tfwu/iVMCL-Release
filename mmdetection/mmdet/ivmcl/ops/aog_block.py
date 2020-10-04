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

import math
import collections

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      build_activation_layer,
                      constant_init, kaiming_init,
                      AttnBatchNorm2d, AttnGroupNorm)
from ..aog import NodeType

class AOGBlock(nn.Module):
    def __init__(self,
                 stage_id,
                 block_id,
                 aog,
                 op_t_node,
                 op_and_node,
                 op_or_node,
                 inplanes,
                 outplanes,
                 bn_ratio=0.25,
                 t_node_no_slice=False,
                 t_node_handle_dblcnt=False,
                 non_t_node_handle_dblcnt=True,
                 or_node_reduction='sum',
                 drop_rate=0.,
                 stride=1,
                 dilation=1,
                 with_group_conv=False,
                 base_width=4,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg_ds=dict(type='BN'),
                 norm_cfg1=dict(type='BN'),
                 norm_cfg2=dict(type='BN'),
                 norm_cfg3=dict(type='BN'),
                 norm_cfg_extra=dict(type='BN'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 dcn=None,
                 plugins=None,
                 use_extra_norm_ac_for_block=True):
        super(AOGBlock, self).__init__()

        self.stage_id = stage_id
        self.block_id = block_id
        self.aog = aog
        self.op_t_node = op_t_node
        self.op_and_node = op_and_node
        self.op_or_node = op_or_node
        self.in_channels = inplanes
        self.out_channels = outplanes
        self.bn_ratio = bn_ratio
        self.t_node_no_slice = t_node_no_slice
        self.t_node_handle_dblcnt = t_node_handle_dblcnt
        self.non_t_node_handle_dblcnt = non_t_node_handle_dblcnt
        self.or_node_reduction = or_node_reduction
        self.drop_rate = drop_rate
        self.stride = stride
        self.dilation = dilation
        self.with_group_conv = with_group_conv
        self.base_width = base_width
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg_ds = norm_cfg_ds
        self.norm_cfg1 = norm_cfg1
        self.norm_cfg2 = norm_cfg2
        self.norm_cfg3 = norm_cfg3
        self.norm_cfg_extra = norm_cfg_extra
        self.act_cfg = act_cfg
        self.act_cfg = act_cfg
        self.dcn = dcn
        self.plugins = plugins
        self.use_extra_norm_ac_for_block = use_extra_norm_ac_for_block

        self.dim = aog.param.grid_wd
        self.in_slices = self._calculate_slices(self.dim, self.in_channels)
        self.out_slices = self._calculate_slices(self.dim, self.out_channels)

        self.node_set = aog.node_set
        self.primitive_set = aog.primitive_set
        self.BFS = aog.BFS
        self.DFS = aog.DFS

        self.hasLateral = {}
        self.hasDblCnt = {}

        self.primitiveDblCnt = None
        self._set_primitive_dbl_cnt()

        self._set_weights_attr()

        self.extra_norm_ac = None
        if self.use_extra_norm_ac_for_block:
            self.extra_norm_ac = self._extra_norm_ac(
                self.norm_cfg_extra,
                self.out_channels)

        self.init_weights()

    def _calculate_slices(self, dim, channels):
        slices = [0] * dim
        for i in range(channels):
            slices[i % dim] += 1
        for d in range(1, dim):
            slices[d] += slices[d - 1]
        slices = [0] + slices
        return slices

    def _set_primitive_dbl_cnt(self):
        self.primitiveDblCnt = [0.0 for i in range(self.dim)]
        for id_ in self.DFS:
            node = self.node_set[id_]
            arr = self.primitive_set[node.rect_idx]
            if node.node_type == NodeType.TerminalNode:
                for i in range(arr.x1, arr.x2+1):
                    self.primitiveDblCnt[i] += node.npaths
        for i in range(self.dim):
            assert self.primitiveDblCnt[i] >= 1.0

    def _set_weights_attr(self):
        for id_ in self.DFS:
            node = self.node_set[id_]
            arr = self.primitive_set[node.rect_idx]
            keep_norm_base = arr.Width() < self.dim
            if keep_norm_base:
                norm_cfg2_ = self.norm_cfg1
            else:
                norm_cfg2_ = self.norm_cfg2

            if node.node_type == NodeType.TerminalNode:
                self.hasLateral[node.id] = False
                self.hasDblCnt[node.id] = False

                inplanes = self.in_channels if self.t_node_no_slice else \
                    self.in_slices[arr.x2 + 1] - self.in_slices[arr.x1]
                outplanes = self.out_slices[arr.x2 +
                                            1] - self.out_slices[arr.x1]
                planes = math.floor(outplanes * self.bn_ratio+0.5)
                stride = self.stride
                groups = 1
                base_channels = planes
                if self.with_group_conv == 1:  # 32x4d
                    groups = math.floor(
                        planes / self.base_width / self.bn_ratio / 2)
                elif self.with_group_conv == 2:  # 64x4d
                    groups = math.floor(
                        planes / self.base_width / self.bn_ratio)
                elif self.with_group_conv == 3:
                    groups = math.floor(planes / self.base_width)

                downsample = None
                if stride != 1 or inplanes != outplanes:
                    if stride > 1:
                        downsample = nn.Sequential(
                            nn.AvgPool2d(kernel_size=(stride, stride),
                                         stride=stride),
                            ConvModule(inplanes,
                                       outplanes,
                                       1,
                                       bias=False,
                                       conv_cfg=self.conv_cfg,
                                       norm_cfg=self.norm_cfg_ds,
                                       act_cfg=None)
                        )
                    else:
                        downsample = ConvModule(inplanes,
                                                outplanes,
                                                1,
                                                bias=False,
                                                conv_cfg=self.conv_cfg,
                                                norm_cfg=self.norm_cfg_ds,
                                                act_cfg=None)

                setattr(self, 'stage_{}_block_{}_node_{}_op'.format(
                    self.stage_id, self.block_id, node.id),
                    self.op_t_node(
                        inplanes=inplanes,
                        planes=planes,
                        outplanes=outplanes,
                        drop_rate=self.drop_rate,
                        stride=stride,
                        dilation=self.dilation,
                        downsample=downsample,
                        groups=groups,
                        base_width=self.base_width,
                        base_channels=base_channels,
                        style=self.style,
                        with_cp=self.with_cp,
                        conv_cfg=self.conv_cfg,
                        norm_cfg1=self.norm_cfg1,
                        norm_cfg2=norm_cfg2_,
                        norm_cfg3=self.norm_cfg3,
                        dcn=self.dcn ))

            elif node.node_type == NodeType.AndNode:
                self.hasLateral[node.id] = False
                self.hasDblCnt[node.id] = False
                for chid in node.child_ids:
                    ch_arr = self.primitive_set[self.node_set[chid].rect_idx]
                    if arr.Width() == ch_arr.Width():
                        self.hasLateral[node.id] = True
                        break
                if self.non_t_node_handle_dblcnt:
                    for chid in node.child_ids:
                        ch_arr = self.primitive_set[
                            self.node_set[chid].rect_idx]
                        if arr.Width() > ch_arr.Width():
                            if node.npaths / self.node_set[chid].npaths != 1.0:
                                self.hasDblCnt[node.id] = True
                                break

                inplanes = self.out_slices[arr.x2 +
                                           1] - self.out_slices[arr.x1]
                outplanes = inplanes
                planes = math.floor(outplanes * self.bn_ratio+0.5)
                stride = 1
                groups = 1
                base_channels = planes
                if self.with_group_conv == 1:  # 32x4d
                    groups = math.floor(
                        planes / self.base_width / self.bn_ratio / 2)
                elif self.with_group_conv == 2:  # 64x4d
                    groups = math.floor(
                        planes / self.base_width / self.bn_ratio)
                elif self.with_group_conv == 3:
                    groups = math.floor(planes / self.base_width)

                setattr(self, 'stage_{}_block_{}_node_{}_op'.format(
                    self.stage_id, self.block_id, node.id),
                    self.op_and_node(
                        inplanes=inplanes,
                        planes=planes,
                        outplanes=outplanes,
                        drop_rate=self.drop_rate,
                        stride=stride,
                        dilation=self.dilation,
                        groups=groups,
                        base_width=self.base_width,
                        base_channels=base_channels,
                        style=self.style,
                        with_cp=self.with_cp,
                        conv_cfg=self.conv_cfg,
                        norm_cfg1=self.norm_cfg1,
                        norm_cfg2=norm_cfg2_,
                        norm_cfg3=self.norm_cfg3,
                        dcn=self.dcn ))

            elif node.node_type == NodeType.OrNode:
                assert self.node_set[node.child_ids[0]].node_type != \
                    NodeType.OrNode

                self.hasLateral[node.id] = False
                self.hasDblCnt[node.id] = False
                for chid in node.child_ids:
                    ch_arr = self.primitive_set[self.node_set[chid].rect_idx]
                    if self.node_set[chid].node_type == NodeType.OrNode or \
                            arr.Width() < ch_arr.Width():
                        self.hasLateral[node.id] = True
                        break
                if self.non_t_node_handle_dblcnt:
                    for chid in node.child_ids:
                        ch_arr = self.primitive_set[
                            self.node_set[chid].rect_idx]
                        if not (self.node_set[chid].node_type ==
                                NodeType.OrNode or arr.Width() < ch_arr.Width()):
                            if node.npaths / self.node_set[chid].npaths != 1.0:
                                self.hasDblCnt[node.id] = True
                                break

                inplanes = self.out_slices[arr.x2 +
                                           1] - self.out_slices[arr.x1]
                outplanes = inplanes
                planes = math.floor(outplanes * self.bn_ratio+0.5)
                stride = 1
                groups = 1
                base_channels = planes
                if self.with_group_conv == 1: # 32x4d
                    groups = math.floor(
                        planes / self.base_width / self.bn_ratio / 2)
                elif self.with_group_conv == 2: # 64x4d
                    groups = math.floor(
                        planes / self.base_width / self.bn_ratio)
                elif self.with_group_conv == 3:
                    groups = math.floor(planes / self.base_width)

                setattr(self, 'stage_{}_block_{}_node_{}_op'.format(
                    self.stage_id, self.block_id, node.id),
                    self.op_or_node(
                        inplanes=inplanes,
                        planes=planes,
                        outplanes=outplanes,
                        drop_rate=self.drop_rate,
                        stride=stride,
                        dilation=self.dilation,
                        groups=groups,
                        base_width=self.base_width,
                        base_channels=base_channels,
                        style=self.style,
                        with_cp=self.with_cp,
                        conv_cfg=self.conv_cfg,
                        norm_cfg1=self.norm_cfg1,
                        norm_cfg2=norm_cfg2_,
                        norm_cfg3=self.norm_cfg3,
                        dcn=self.dcn ))

    def _extra_norm_ac(self, norm_cfg, num_features):
        return nn.Sequential(
            build_norm_layer(norm_cfg, num_features)[1],
            build_activation_layer(self.act_cfg))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (AttnBatchNorm2d, AttnGroupNorm)):
                nn.init.normal_(m.weight_, 1., 0.1)
                nn.init.normal_(m.bias_, 0., 0.1)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, x):
        NodeIdTensorDict = {}

        # handle input x
        tnode_dblcnt = False
        if self.t_node_handle_dblcnt and self.in_channels == self.out_channels:
            x_scaled = []
            for i in range(self.dim):
                left, right = self.in_slices[i], self.in_slices[i+1]
                x_scaled.append(x[:, left:right, :, :].div(
                    self.primitiveDblCnt[i]))
            xx = torch.cat(x_scaled, 1)
            tnode_dblcnt = True

        DFS_ = self.DFS

        # T-nodes
        for id_ in DFS_:
            node = self.node_set[id_]
            op_name = 'stage_{}_block_{}_node_{}_op'.format(
                self.stage_id, self.block_id, node.id)

            if node.node_type == NodeType.TerminalNode:
                arr = self.primitive_set[node.rect_idx]
                right, left = self.in_slices[arr.x2 +
                                             1], self.in_slices[arr.x1]
                tnode_tensor_op = x if self.t_node_no_slice else \
                    x[:, left:right, :, :].contiguous()  # TODO: use unfold ?
                # assert tnode_tensor.requires_grad, 'slice needs to retain grad'

                if tnode_dblcnt:
                    tnode_tensor_res = xx[:, left:right, :, :].mul(node.npaths)
                    tnode_output = getattr(self, op_name)(
                        tnode_tensor_op,  identity=tnode_tensor_res)
                else:
                    tnode_output = getattr(self, op_name)(tnode_tensor_op)

                NodeIdTensorDict[node.id] = tnode_output

        # AND- and OR-nodes
        node_op_idx_ = 0
        for id_ in DFS_:
            node = self.node_set[id_]
            arr = self.primitive_set[node.rect_idx]
            op_name = 'stage_{}_block_{}_node_{}_op'.format(
                self.stage_id, self.block_id, node.id)

            if node.node_type == NodeType.AndNode:
                if self.hasDblCnt[node.id]:
                    child_tensor_res = []
                    child_tensor_op = []
                    for chid in node.child_ids:
                        if chid not in DFS_:
                            continue
                        ch_arr = self.primitive_set[self.node_set[chid].rect_idx]
                        if arr.Width() > ch_arr.Width():
                            factor = node.npaths / self.node_set[chid].npaths
                            if factor == 1.0:
                                child_tensor_res.append(NodeIdTensorDict[chid])
                            else:
                                child_tensor_res.append(
                                    NodeIdTensorDict[chid].mul(factor))
                            child_tensor_op.append(NodeIdTensorDict[chid])

                    anode_tensor_res = torch.cat(child_tensor_res, 1)
                    anode_tensor_op = torch.cat(child_tensor_op, 1)

                    if self.hasLateral[node.id]:
                        ids1 = set(node.parent_ids)
                        num_shared = 0
                        for chid in node.child_ids:
                            if chid not in DFS_:
                                continue
                            ch_arr = self.primitive_set[
                                self.node_set[chid].rect_idx]
                            ids2 = self.node_set[chid].parent_ids
                            if arr.Width() == ch_arr.Width():
                                anode_tensor_op = anode_tensor_op + \
                                    NodeIdTensorDict[chid]
                                if len(ids1.intersection(ids2)) == num_shared:
                                    anode_tensor_res = anode_tensor_res + \
                                        NodeIdTensorDict[chid]

                    anode_output = getattr(self, op_name)(
                        anode_tensor_op, identity=anode_tensor_res)

                else:
                    child_tensor = []
                    for chid in node.child_ids:
                        if chid not in DFS_:
                            continue
                        ch_arr = self.primitive_set[
                            self.node_set[chid].rect_idx]
                        if arr.Width() > ch_arr.Width():
                            child_tensor.append(NodeIdTensorDict[chid])

                    anode_tensor_op = torch.cat(child_tensor, 1)

                    if self.hasLateral[node.id]:
                        ids1 = set(node.parent_ids)
                        num_shared = 0
                        for chid in node.child_ids:
                            if chid not in DFS_:
                                continue
                            ch_arr = self.primitive_set[
                                self.node_set[chid].rect_idx]
                            ids2 = self.node_set[chid].parent_ids
                            if arr.Width() == ch_arr.Width() and \
                                    len(ids1.intersection(ids2)) == num_shared:
                                anode_tensor_op = anode_tensor_op + \
                                    NodeIdTensorDict[chid]

                        anode_tensor_res = anode_tensor_op

                        for chid in node.child_ids:
                            if chid not in DFS_:
                                continue
                            ch_arr = self.primitive_set[
                                self.node_set[chid].rect_idx]
                            ids2 = self.node_set[chid].parent_ids
                            if arr.Width() == ch_arr.Width() and \
                                    len(ids1.intersection(ids2)) > num_shared:
                                anode_tensor_op = anode_tensor_op + \
                                    NodeIdTensorDict[chid]

                        anode_output = getattr(self, op_name)(
                            anode_tensor_op, identity=anode_tensor_res)
                    else:
                        anode_output = getattr(self, op_name)(anode_tensor_op)

                NodeIdTensorDict[node.id] = anode_output

            elif node.node_type == NodeType.OrNode:
                num_op_sum = 0.
                num_res_sum = 0.
                if self.hasDblCnt[node.id]:
                    factor = node.npaths / \
                        self.node_set[node.child_ids[0]].npaths

                    if factor == 1.0:
                        onode_tensor_res = NodeIdTensorDict[node.child_ids[0]]
                    else:
                        onode_tensor_res = \
                            NodeIdTensorDict[node.child_ids[0]].mul(factor)
                    num_res_sum += 1.

                    onode_tensor_op = NodeIdTensorDict[node.child_ids[0]]
                    num_op_sum += 1.
                    for chid in node.child_ids[1:]:
                        if chid not in DFS_:
                            continue
                        if self.node_set[chid].node_type != NodeType.OrNode:
                            ch_arr = self.primitive_set[
                                self.node_set[chid].rect_idx]
                            if arr.Width() == ch_arr.Width():
                                factor = node.npaths / \
                                    self.node_set[chid].npaths
                                if factor == 1.0:
                                    onode_tensor_res = onode_tensor_res + \
                                        NodeIdTensorDict[chid]
                                else:
                                    onode_tensor_res = onode_tensor_res + \
                                        NodeIdTensorDict[chid].mul(factor)
                                num_res_sum += 1.
                                if self.or_node_reduction == 'max':
                                    onode_tensor_op = torch.max(
                                        onode_tensor_op, NodeIdTensorDict[chid])
                                else:
                                    onode_tensor_op = onode_tensor_op + \
                                        NodeIdTensorDict[chid]
                                    num_op_sum += 1.

                    if self.hasLateral[node.id]:
                        ids1 = set(node.parent_ids)
                        num_shared = 0
                        for chid in node.child_ids[1:]:
                            if chid not in DFS_:
                                continue
                            ids2 = self.node_set[chid].parent_ids
                            if self.node_set[chid].node_type == \
                                    NodeType.OrNode and \
                                    len(ids1.intersection(ids2)) == num_shared:
                                onode_tensor_res = onode_tensor_res + \
                                    NodeIdTensorDict[chid]
                                num_res_sum += 1.
                                if self.or_node_reduction == 'max':
                                    onode_tensor_op = torch.max(
                                        onode_tensor_op, NodeIdTensorDict[chid])
                                else:
                                    onode_tensor_op = onode_tensor_op + \
                                        NodeIdTensorDict[chid]
                                    num_op_sum += 1.

                        for chid in node.child_ids[1:]:
                            if chid not in DFS_:
                                continue
                            ch_arr = \
                                self.primitive_set[self.node_set[chid].rect_idx]
                            ids2 = self.node_set[chid].parent_ids
                            if self.node_set[chid].node_type == \
                                NodeType.OrNode and \
                                    len(ids1.intersection(ids2)) > num_shared:

                                if self.or_node_reduction == 'max':
                                    onode_tensor_op = torch.max(
                                        onode_tensor_op, NodeIdTensorDict[chid])
                                else:
                                    onode_tensor_op = onode_tensor_op + \
                                        NodeIdTensorDict[chid]
                                    num_op_sum += 1.
                            elif self.node_set[chid].node_type == \
                                    NodeType.TerminalNode and \
                                    arr.Width() < ch_arr.Width():
                                ch_left = self.out_slices[arr.x1] - \
                                    self.out_slices[ch_arr.x1]
                                ch_right = self.out_slices[arr.x2 + 1] - \
                                    self.out_slices[ch_arr.x1]
                                if self.or_node_reduction == 'max':
                                    onode_tensor_op = torch.max(
                                        onode_tensor_op,
                                        NodeIdTensorDict[chid][:, ch_left:ch_right, :, :])
                                else:
                                    onode_tensor_op = onode_tensor_op + \
                                        NodeIdTensorDict[chid][:,
                                                               ch_left:ch_right, :, :]  # .contiguous()
                                    num_op_sum += 1.

                    if self.or_node_reduction == 'avg':
                        onode_tensor_res = onode_tensor_res / num_res_sum
                        onode_tensor_op = onode_tensor_op / num_op_sum

                    onode_output = getattr(self, op_name)(
                        onode_tensor_op, identity=onode_tensor_res)
                else:
                    if self.or_node_reduction == 'max':
                        onode_tensor_op = NodeIdTensorDict[node.child_ids[0]]
                        onode_tensor_res = NodeIdTensorDict[node.child_ids[0]]
                        for chid in node.child_ids[1:]:
                            if chid not in DFS_:
                                continue
                            if self.node_set[chid].node_type != NodeType.OrNode:
                                ch_arr = \
                                    self.primitive_set[self.node_set[chid].rect_idx]
                                if arr.Width() == ch_arr.Width():
                                    onode_tensor_op = torch.max(
                                        onode_tensor_op, NodeIdTensorDict[chid])
                                    onode_tensor_res = onode_tensor_res + \
                                        NodeIdTensorDict[chid]

                        if self.hasLateral[node.id]:
                            ids1 = set(node.parent_ids)
                            num_shared = 0
                            for chid in node.child_ids[1:]:
                                if chid not in DFS_:
                                    continue
                                ids2 = self.node_set[chid].parent_ids
                                if self.node_set[chid].node_type == \
                                        NodeType.OrNode and \
                                        len(ids1.intersection(ids2)) == num_shared:
                                    onode_tensor_op = torch.max(
                                        onode_tensor_op, NodeIdTensorDict[chid])
                                    onode_tensor_res = onode_tensor_res + \
                                        NodeIdTensorDict[chid]

                            for chid in node.child_ids[1:]:
                                if chid not in DFS_:
                                    continue
                                ch_arr = \
                                    self.primitive_set[self.node_set[chid].rect_idx]
                                ids2 = self.node_set[chid].parent_ids
                                if self.node_set[chid].node_type == NodeType.OrNode and \
                                        len(ids1.intersection(ids2)) > num_shared:
                                    onode_tensor_op = torch.max(
                                        onode_tensor_op, NodeIdTensorDict[chid])
                                elif self.node_set[chid].node_type == NodeType.TerminalNode and \
                                        arr.Width() < ch_arr.Width():
                                    ch_left = self.out_slices[arr.x1] - \
                                        self.out_slices[ch_arr.x1]
                                    ch_right = self.out_slices[arr.x2 +
                                                               1] - self.out_slices[ch_arr.x1]
                                    onode_tensor_op = torch.max(
                                        onode_tensor_op,
                                        NodeIdTensorDict[chid][:, ch_left:ch_right, :, :])

                            onode_output = getattr(self, op_name)(
                                onode_tensor_op, identity=onode_tensor_res)
                        else:
                            onode_output = getattr(self, op_name)(onode_tensor_op)
                    else:
                        onode_tensor_op = NodeIdTensorDict[node.child_ids[0]]
                        num_op_sum += 1.
                        for chid in node.child_ids[1:]:
                            if chid not in DFS_:
                                continue
                            if self.node_set[chid].node_type != NodeType.OrNode:
                                ch_arr = self.primitive_set[self.node_set[chid].rect_idx]
                                if arr.Width() == ch_arr.Width():
                                    onode_tensor_op = onode_tensor_op + \
                                        NodeIdTensorDict[chid]
                                    num_op_sum += 1.

                        if self.hasLateral[node.id]:
                            ids1 = set(node.parent_ids)
                            num_shared = 0
                            for chid in node.child_ids[1:]:
                                if chid not in DFS_:
                                    continue
                                ids2 = self.node_set[chid].parent_ids
                                if self.node_set[chid].node_type == NodeType.OrNode and \
                                        len(ids1.intersection(ids2)) == num_shared:
                                    onode_tensor_op = onode_tensor_op + \
                                        NodeIdTensorDict[chid]
                                    num_op_sum += 1.

                            onode_tensor_res = onode_tensor_op
                            num_res_sum = num_op_sum

                            for chid in node.child_ids[1:]:
                                if chid not in DFS_:
                                    continue
                                ch_arr = self.primitive_set[self.node_set[chid].rect_idx]
                                ids2 = self.node_set[chid].parent_ids
                                if self.node_set[chid].node_type == NodeType.OrNode and \
                                        len(ids1.intersection(ids2)) > num_shared:
                                    onode_tensor_op = onode_tensor_op + \
                                        NodeIdTensorDict[chid]
                                    num_op_sum += 1.
                                elif self.node_set[chid].node_type == NodeType.TerminalNode and \
                                        arr.Width() < ch_arr.Width():
                                    ch_left = self.out_slices[arr.x1] - \
                                        self.out_slices[ch_arr.x1]
                                    ch_right = self.out_slices[arr.x2 +
                                                               1] - self.out_slices[ch_arr.x1]
                                    onode_tensor_op = onode_tensor_op + \
                                        NodeIdTensorDict[chid][:, ch_left:ch_right, :, :].contiguous(
                                        )
                                    num_op_sum += 1.

                            if self.or_node_reduction == 'avg':
                                onode_tensor_op = onode_tensor_op / num_op_sum
                                onode_tensor_res = onode_tensor_res / num_res_sum

                            onode_output = getattr(self, op_name)(
                                onode_tensor_op, identity=onode_tensor_res)
                        else:
                            if self.or_node_reduction == 'avg':
                                onode_tensor_op = onode_tensor_op / num_op_sum
                            onode_output = getattr(
                                self, op_name)(onode_tensor_op)

                NodeIdTensorDict[node.id] = onode_output

        out = NodeIdTensorDict[self.aog.BFS[0]]

        if self.extra_norm_ac is not None:
            out = self.extra_norm_ac(out)

        return out
