# reference: https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/env.py

import torch


TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
"""
PyTorch version as a tuple of 2 ints. Useful for comparizon.
"""