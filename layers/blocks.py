# reference: https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/blocks.py

from ..core.nn import weight_init
from torch import nn

from .batch_norm import FrozenBatchNorm2d


"""
CNN building blocks.
"""


class CNNBlockBase(nn.Module):
    """
    A CNN block is assumed to have input channels, output channels and a stride.
    The input and output of `forward()` method must be NCHW tensors.
    The method can perform arbitrary computation but must match the given
    channels and stride specification.
    """

    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these argments.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
    
    def freeze(self):
        """
        Make this block not trainable.
        This method sets all parameters to `requires_grad=False`,
        and convert all BatchNorm layers to FrozenBatchNorm
        """
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.covert_frozen_batchnorm(self)
        return self