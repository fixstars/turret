# -*- coding: utf-8 -*-
from .builtin import padding


def pad(input, pad_width):
    """Layer for padding.

    Args:
        input(turret.Tensor): The tensor which will be processed by layer.
        pad_width(float): The padding that is applied at the tensor.

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    return padding(input, pad_width, pad_width)
