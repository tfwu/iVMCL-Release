import torch.nn as nn

from .registry import PLUGIN_LAYERS


@PLUGIN_LAYERS.register_module()
class Dropout2d_(nn.Dropout2d):
    """ To fit the plugin interface
    """

    def __init__(self, in_channels, p, inplace=True):
        super(Dropout2d_, self).__init__(p, inplace=inplace)


@PLUGIN_LAYERS.register_module()
class SELayer(nn.Module):
    _abbr_ = 'se'
    def __init__(self, in_channels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
