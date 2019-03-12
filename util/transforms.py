import torch
from torchvision import datasets, transforms
import pdb

# TODO: Verify this is necessary


class ToDouble(object):
    def __call__(self, pic):
        return pic.double()


class Identity(object):
    def __call__(self, pic):
        return pic
