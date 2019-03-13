# -*- coding: utf-8 -*-
from .builtin import ActivationType
from .builtin import builtin_activation


def relu(input):
    """Layer for ReLU.

    Args:
        input(turret.Tensor): Tensor which will be processed by ReLU.

    Returns:
        tensor(turret.Tensor): Tensor processed by ReLU.
    """
    return builtin_activation(input, ActivationType.RELU)
