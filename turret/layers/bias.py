# -*- coding: utf-8 -*-
from .builtin import scale
from .builtin import ScaleMode


def bias(input, b):
    """Layer to bias.

    Args:
        input(turret.Tensor): Tensor which will be biased.
        b(np.ndarray): Bias.

    Returns:
        tensor(turret.Tensor): Biased tensor.
    """
    c = input.dimensions[0].size
    if b.shape != (c,):
        raise ValueError("shape mismatch")
    return scale(input, ScaleMode.CHANNEL, shift=b)
