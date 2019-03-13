# -*- coding: utf-8 -*-
from .builtin import shuffle_and_reshape


def reshape(input, shape):
    """Layer to reshape only.

    Args:
        input(turret.Tensor): The tensor which will be reshaped.
        shape(turret.Dimensions): Reshaped dimensions.

    Returns:
        tensor(turret.Tensor): Reshaped tensor.
    """
    return shuffle_and_reshape(input, None, shape, None)
