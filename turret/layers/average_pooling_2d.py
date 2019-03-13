# -*- coding: utf-8 -*-
from .builtin import PoolingType
from .builtin import builtin_pooling


def average_pooling_2d(input, window_size, stride=1, padding=0):
    """Layer for average 2D pooling.

    Args:
        input(turret.Tensor): Tensor which will be processed by pooling.
        window_size(tupple): Window size for pooling.
        stride(int): Stride for pooling.
        padding: Padding for pooling.

    Returns:
        tensor(turret.Tensor): Tensor processed by pooling.
    """
    return builtin_pooling(input, PoolingType.AVERAGE,
                           window_size, stride, padding)
