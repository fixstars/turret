# -*- coding: utf-8 -*-
from .builtin import PoolingType
from .builtin import builtin_pooling


def max_average_blend_pooling_2d(input, window_size, stride=1, padding=0,
                                 blend=0.0):
    """Layer for 2D pooling(max average blend mode).

    Args:
        input(turret.Tensor): Tensor which will be processed by pooling.
        window_size(tupple): Window size for pooling.
        stride(int): Stride for pooling.
        padding: Padding for pooling.
        blend(float): Blending factor for the max_average_blend mode:
            max_average_blendPool = (1-blendFactor)*maxPool + blendFactor*avgPool
            blendFactor is a user value in [0,1] with the default value of 0.0

    Returns:
        tensor(turret.Tensor): Tensor processed by pooling.
    """
    return builtin_pooling(input, PoolingType.MAX_AVERAGE_BLEND,
                           window_size, stride, padding, blend)
