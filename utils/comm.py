# reference: https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/comm.py

import torch.distributed as dist


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()