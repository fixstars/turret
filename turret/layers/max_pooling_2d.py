# -*- coding: utf-8 -*-
from .builtin import PoolingType
from .builtin import builtin_pooling


def max_pooling_2d(input, window_size, stride=1, padding=0):
    """Layer for 2D pooling(max mode).

    Args:
        input(turret.Tensor): Tensor which will be processed by pooling.
        window_size(tupple): Window size for pooling.
        stride(int): Stride for pooling.
        padding: Padding for pooling.

    Returns:
        tensor(turret.Tensor): Tensor processed by pooling.
    """
    return builtin_pooling(input, PoolingType.MAX,
                           window_size, stride, padding)
