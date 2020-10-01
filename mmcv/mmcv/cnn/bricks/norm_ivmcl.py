import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import build_activation_layer


class AttnWeights(nn.Module):
    """ Attention weights for the mixture of affine transformations
        https://arxiv.org/abs/1908.01259
    """
    def __init__(self,
                 attn_mode,
                 num_features,
                 num_affine_trans,
                 num_groups=0,
                 use_rsd=True,
                 use_maxpool=False,
                 eps=1e-3,
                 act_cfg=dict(type="HSigmoidv2")):
        super(AttnWeights, self).__init__()

        if use_rsd:
            use_maxpool = False

        self.num_affine_trans = num_affine_trans
        self.use_rsd = use_rsd
        self.use_maxpool = use_maxpool
        self.eps = eps
        if not self.use_rsd:
            self.avgpool = nn.AdaptiveAvgPool2d(1)

        layers = []
        if attn_mode == 0:
            layers = [nn.Conv2d(num_features, num_affine_trans, 1, bias=False),
                      nn.BatchNorm2d(num_affine_trans),
                      build_activation_layer(act_cfg)]
        elif attn_mode == 1:
            assert num_groups > 0
            layers = [nn.Conv2d(num_features, num_affine_trans, 1, bias=False),
                      nn.GroupNorm(num_channels=num_affine_trans,
                                   num_groups=num_groups),
                      build_activation_layer(act_cfg)]
        else:
            raise NotImplementedError("Unknow attention weight type")

        self.attention = nn.Sequential(*layers)

    def forward(self, x):
        b, c, h, w = x.size()
        if self.use_rsd:
            var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True)
            y = mean * (var + self.eps).rsqrt()

            # var = torch.var(x, dim=(2, 3), keepdim=True)
            # y *= (var + self.eps).rsqrt()
        else:
            y = self.avgpool(x)
            if self.use_maxpool:
                y += F.max_pool2d(x, (h, w), stride=(h, w)).view(b, c, 1, 1)
        return self.attention(y).view(b, self.num_affine_trans)


class AttnBatchNorm2d(nn.BatchNorm2d):
    """ Attentive Normalization with BatchNorm2d backbone
        https://arxiv.org/abs/1908.01259
    """

    _abbr_ = "AttnBN2d"

    def __init__(self,
                 num_features,
                 num_affine_trans,
                 attn_mode=0,
                 eps=1e-5,
                 momentum=0.1,
                 track_running_stats=True,
                 use_rsd=True,
                 use_maxpool=False,
                 eps_var=1e-3,
                 act_cfg=dict(type="HSigmoidv2")):
        super(AttnBatchNorm2d, self).__init__(num_features,
                                              affine=False,
                                              eps=eps,
                                              momentum=momentum,
                                              track_running_stats=track_running_stats)

        self.num_affine_trans = num_affine_trans
        self.attn_mode = attn_mode
        self.use_rsd = use_rsd
        self.eps_var = eps_var
        self.act_cfg = act_cfg

        self.weight_ = nn.Parameter(
            torch.Tensor(num_affine_trans, num_features))
        self.bias_ = nn.Parameter(
            torch.Tensor(num_affine_trans, num_features))

        self.attn_weights = AttnWeights(attn_mode,
                                        num_features,
                                        num_affine_trans,
                                        use_rsd=use_rsd,
                                        use_maxpool=use_maxpool,
                                        eps=eps_var,
                                        act_cfg=act_cfg)

        self._init_params()

    def _init_params(self):
        nn.init.normal_(self.weight_, 1., 0.1)
        nn.init.normal_(self.bias_, 0., 0.1)

    def forward(self, x):
        output = super(AttnBatchNorm2d, self).forward(x)
        size = output.size()
        y = self.attn_weights(x)  # bxk

        weight = y @ self.weight_  # bxc
        bias = y @ self.bias_  # bxc
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)

        return weight * output + bias


class AttnGroupNorm(nn.Module):
    """Attentive Normalization with GroupNorm backbone
        https://arxiv.org/abs/1908.01259
    """
    __constants__ = ['num_groups', 'num_features', 'num_affine_trans', 'eps',
                     'weight', 'bias']
    _abbr_ = "AttnGN"

    def __init__(self,
                 num_features,
                 num_affine_trans,
                 num_groups,
                 attn_mode=1,
                 eps=1e-5,
                 use_rsd=True,
                 use_maxpool=False,
                 eps_var=1e-3,
                 act_cfg=dict(type="HSigmoidv2")):
        super(AttnGroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        self.num_affine_trans = num_affine_trans
        self.eps = eps
        self.affine = True
        self.weight_ = nn.Parameter(
            torch.Tensor(num_affine_trans, num_features))
        self.bias_ = nn.Parameter(
            torch.Tensor(num_affine_trans, num_features))

        self.register_parameter('weight', None)
        self.register_parameter('bias', None)

        self.attn_weights = AttnWeights(attn_mode,
                                        num_features,
                                        num_affine_trans,
                                        num_groups=num_affine_trans//2,
                                        use_rsd=use_rsd,
                                        use_maxpool=use_maxpool,
                                        eps=eps_var,
                                        act_cfg=act_cfg)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight_, 1., 0.1)
        nn.init.normal_(self.bias_, 0., 0.1)

    def forward(self, x):
        output = F.group_norm(
            x, self.num_groups, self.weight, self.bias, self.eps)
        size = output.size()

        y = self.attention_weights(x)

        weight = y @ self.weight_
        bias = y @ self.bias_

        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)

        return weight * output + bias

    def extra_repr(self):
        return '{num_groups}, {num_features}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)


