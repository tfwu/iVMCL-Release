import torch.nn as nn
import torch.nn.functional as F

from .registry import ACTIVATION_LAYERS


@ACTIVATION_LAYERS.register_module()
class HSigmoidv2(nn.Module):
    """ (add ref)
    """
    def __init__(self, inplace=True):
        super(HSigmoidv2, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3., inplace=self.inplace) / 6.
        return out

