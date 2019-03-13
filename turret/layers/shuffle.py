# -*- coding: utf-8 -*-
from .builtin import shuffle_and_reshape
from ..foundational import Dimension
from ..foundational import Dimensions

def shuffle(input, order):
    """Layer to shuffle.

    Args:
        input(turret.Tensor): The tensor which will be processed by layer.
        order(tuple): The permutation applied by the first transpose
            operation.

    Returns:
        tensor(turret.Tensor): Tensor processed by layer.
    """
    # IShuffleLayer shuffles also the types of dimensions.
    # turret.shuffle keeps the types of dimensions.
    in_dims = input.dimensions
    out_dims = Dimensions([
        Dimension(in_dims[j].size, in_dims[i].type)
        for i, j in enumerate(order)])
    return shuffle_and_reshape(input, order, out_dims, None)
