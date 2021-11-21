# FasterRCNN

This repository is the minimum copy of detectron2 for understanding pytorch and object detection models.

## Setup
As a runtime, I used `nvcr.io/nvidia/pytorch:21.10-py3` container.

And I should setup c extensions.
```shell
cd lyaers/csrc
python setup.py install
```