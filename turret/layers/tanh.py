# -*- coding: utf-8 -*-
from .builtin import ActivationType
from .builtin import builtin_activation


def tanh(input):
    """Layer for tanh.

    Args:
        input(turret.Tensor): Tensor which will be processed by tanh.

    Returns:
        tensor(turret.Tensor): Tensor processed by tanh.
    """
    return builtin_activation(input, ActivationType.TANH)
