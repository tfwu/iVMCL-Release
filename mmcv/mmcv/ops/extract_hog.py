import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', [
    'extract_hog_forward'])


class ExtractHOGFunction(Function):

    @staticmethod
    def symbolic(g, input, sbin):
        return g.op(
            'MMCVExtractHOG',
            input,
            sbin)

    @staticmethod
    def forward(ctx, input, sbin):
        assert input.is_cuda and input.size(1) == 3
        assert isinstance(sbin, int)

        blocks_height = int(round(float(input.size(2)) / float(sbin)))
        blocks_width = int(round(float(input.size(3)) / float(sbin)))
        visible_height = blocks_height * sbin
        visible_width = blocks_width * sbin

        grad_size = (input.size(0), 1, visible_height-2, visible_width-2)
        grad_v = input.new_zeros(grad_size, requires_grad=False)
        grad_i = input.new_zeros(grad_size, requires_grad=False)

        hist_channels = 18
        hist = input.new_zeros(
            (input.size(0), hist_channels, blocks_height, blocks_width), requires_grad=False)
        norm = input.new_zeros(
            (input.size(0), 1,             blocks_height, blocks_width), requires_grad=False)

        out_h = max(blocks_height-2, 0)
        out_w = max(blocks_width-2,  0)
        out_channel = 27+4
        out_size = (input.size(0), out_channel, out_h, out_w)
        output = input.new_zeros(out_size, requires_grad=False)

        ext_module.extract_hog_forward(
            input, sbin, output, grad_v, grad_i, hist, norm)

        output = F.pad(input=output,
                       pad=(1, 1, 1, 1, 0, 0, 0, 0),
                       mode='constant', value=0)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


extract_hog = ExtractHOGFunction.apply


class ExtractHOG(nn.Module):

    def __init__(self, sbin=4):
        super(ExtractHOG, self).__init__()

        self.sbin = sbin

    def forward(self, input):
        with torch.no_grad():
            return extract_hog(input, self.sbin)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'sbin={self.sbin}'
        return s
