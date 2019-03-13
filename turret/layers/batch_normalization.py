# -*- coding: utf-8 -*-
import numpy as np

from .builtin import scale
from .builtin import ScaleMode


def batch_normalization(input, mean, variance, gamma=None, beta=None, eps=2e-5):
    """Layer for batch normalization.

    Args:
        input(turret.Tensor): Tensor which will be processed by batch
            normalization.
        mean(turret.Tensor): The initial mean value.
        variance(turret.Tensor): The initial variance value.
        gamma(turret.Tensor): The initial gamma value.
        beta(turret.Tensor): The initial beta value.
        eps(float): Small float added to variance to avoid dividing
            by zero.

    Returns:
        tensor(turret.Tensor): Tensor processed by batch normalization.
    """
    c = input.dimensions[0].size
    if gamma is None:
        gamma = np.ones((c,), dtype=mean.dtype)
    if beta is None:
        beta = np.zeros((c,), dtype=mean.dtype)

    if mean.shape != (c,):
        raise ValueError("shape mismatch")
    if variance.shape != (c,):
        raise ValueError("shape mismatch")
    if gamma.shape != (c,):
        raise ValueError("shape mismatch")
    if beta.shape != (c,):
        raise ValueError("shape mismatch")

    stddev = variance ** 0.5
    sh = beta - gamma * mean / (stddev + eps)
    sc = gamma / (stddev + eps)
    return scale(input, ScaleMode.CHANNEL, shift=sh, scale=sc)
