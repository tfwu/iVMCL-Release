import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import (AttnBatchNorm2d, AttnGroupNorm)

def separate_norm_params(model):
    norm_params = []
    base_params = []
    for name, m in model.named_modules():
        if isinstance(m, (AttnBatchNorm2d, AttnGroupNorm)):
            # print(name)
            for p in m.parameters(False):
                # print(p.shape)
                norm_params.append(p)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            # print(name)
            for p in m.parameters(False):
                # print(p.shape)
                norm_params.append(p)
        else:
            for p in m.parameters(False):
                if p.requires_grad:
                    base_params.append(p)

    assert sum([m.numel() for m in model.parameters()]) == \
        sum([p.numel() for p in norm_params]) + \
            sum([p.numel() for p in base_params])

    return norm_params, base_params


