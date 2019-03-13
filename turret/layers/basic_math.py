# -*- coding: utf-8 -*-
from .builtin import elementwise
from .builtin import unary
from .builtin import ElementWiseOperation
from .builtin import UnaryOperation


def sum(*xs):
    """Get the sum of elements.

    Args:
        xs(tuple): Tuple of tensor.

    Returns:
        tensor(turret.Tensor): Summed tensor.
    """
    assert(len(xs) > 0)
    h = xs[0]
    for x in xs[1:]:
        h = elementwise(h, x, ElementWiseOperation.SUM)
    return h

def prod(*xs):
    """Get the product of elements.

    Args:
        xs(tuple): Tuple of tensor.

    Returns:
        tensor(turret.Tensor): Producted tensor.
    """
    assert(len(xs) > 0)
    h = xs[0]
    for x in xs[1:]:
        h = elementwise(h, x, ElementWiseOperation.PROD)
    return h

def max(*xs):
    """Get the maximum of elements.

    Args:
        xs(tuple): Tuple of tensor.

    Returns:
        tensor(turret.Tensor): Maximum tensor.
    """
    assert(len(xs) > 0)
    h = xs[0]
    for x in xs[1:]:
        h = elementwise(h, x, ElementWiseOperation.MAX)
    return h

def min(*xs):
    """Get the minimum of elements.

    Args:
        xs(tuple): Tuple of tensor.

    Returns:
        tensor(turret.Tensor): Minimum tensor.
    """
    assert(len(xs) > 0)
    h = xs[0]
    for x in xs[1:]:
        h = elementwise(h, x, ElementWiseOperation.MIN)
    return h

def sub(input0, input1):
    """Get the difference of two elements.

    Args:
        input0(turret.Tensor): The first input tensor.
        input1(turret.Tensor): The second input tensor.

    Returns:
        tensor(turret.Tensor): The difference of two elements.
    """
    return elementwise(input0, input1, ElementWiseOperation.SUB)

def div(input0, input1):
    """Get the quotient of two elements.

    Args:
        input0(turret.Tensor): The first input tensor.
        input1(turret.Tensor): The second input tensor.

    Returns:
        tensor(turret.Tensor): The quotient of two elements.
    """
    return elementwise(input0, input1, ElementWiseOperation.DIV)

def pow(input0, input1):
    """Get the power of two elements.

    Args:
        input0(turret.Tensor): The first input tensor.
        input1(turret.Tensor): The second input tensor.

    Returns:
        tensor(turret.Tensor): The power of two elements.
    """
    return elementwise(input0, input1, ElementWiseOperation.POW)


def exp(input):
    """Get the exponent of element.

    Args:
        input(turret.Tensor): The input tensor.

    Returns:
        tensor(turret.Tensor): The exponent of element.
    """
    return unary(input, UnaryOperation.EXP)

def log(input):
    """Get the log(base e) of element.

    Args:
        input(turret.Tensor): The input tensor.

    Returns:
        tensor(turret.Tensor): The log of element.
    """
    return unary(input, UnaryOperation.LOG)

def sqrt(input):
    """Get the square root of element.

    Args:
        input(turret.Tensor): The input tensor.

    Returns:
        tensor(turret.Tensor): The square root of element.
    """
    return unary(input, UnaryOperation.SQRT)

def recip(input):
    """Get the reciprocal of element.

    Args:
        input(turret.Tensor): The input tensor.

    Returns:
        tensor(turret.Tensor): The reciprocal of element.
    """
    return unary(input, UnaryOperation.RECIP)

def abs(input):
    """Get the absolute value of element.

    Args:
        input(turret.Tensor): The input tensor.

    Returns:
        tensor(turret.Tensor): The absolute value of element.
    """
    return unary(input, UnaryOperation.ABS)

def neg(input):
    """Get the negation of element.

    Args:
        input(turret.Tensor): The input tensor.

    Returns:
        tensor(turret.Tensor): The negation of element.
    """
    return unary(input, UnaryOperation.NEG)
