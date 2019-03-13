# -*- coding: utf-8 -*-
import numpy as np

from .builtin import scale
from .builtin import ScaleMode
from .builtin import elementwise
from .builtin import ElementWiseOperation


def leaky_relu(input, slope=0.2):
    """Layer for leaky ReLU.

    Args:
        input(turret.Tensor): Tensor which will be processed by leaky ReLU.
        slope(np.ndarray): Slope value.

    Returns:
        tensor(turret.Tensor): Tensor processed by leaky ReLU.
    """
    s = np.array([slope], dtype=np.float32)
    return elementwise(
        input, scale(input, ScaleMode.UNIFORM, scale=s),
        ElementWiseOperation.MAX)
