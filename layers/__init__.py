# reference: https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/__init__.py

from .batch_norm import FrozenBatchNorm2d, get_norm, NaiveSyncBatchNorm
from .shape_spec import ShapeSpec
from .wrappers import (
    BatchNorm2d,
    Conv2d,
)
from .blocks import CNNBlockBase