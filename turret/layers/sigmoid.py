# -*- coding: utf-8 -*-
from .builtin import ActivationType
from .builtin import builtin_activation


def sigmoid(input):
    """Layer for sigmoid.

    Args:
        input(turret.Tensor): Tensor which will be processed by sigmoid.

    Returns:
        tensor(turret.Tensor): Tensor processed by sigmoid.
    """
    return builtin_activation(input, ActivationType.SIGMOID)
