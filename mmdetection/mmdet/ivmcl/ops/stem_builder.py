import torch.nn as nn

from .stem import Stem, DeepStem

stem_cfg = {
    # layer_abbreviation: module
    'Stem': Stem,
    'DeepStem': DeepStem,
}


def build_stem_layer(cfg):
    """ Build stem layer

    Args:
        cfg (dict): cfg should contain:
            type (str): Identify activation layer type.
            layer args: args needed to instantiate a stem layer.

    Returns:
        layer (nn.Module): Created stem layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in stem_cfg:
        raise KeyError('Unrecognized stem type {}'.format(layer_type))
    else:
        stem = stem_cfg[layer_type]
        if stem is None:
            raise NotImplementedError

    layer = stem(**cfg_)
    return layer
