# -*- coding: utf-8 -*-
import numpy as np

from .foundational import DataType


def get_datatype(dtype):
    """
    Convert the data type of Numpy to it of Turret.

    Args:
        dtype: The data type of Numpy.

    Returns:
        turret.DataType: The data type of Turret.
    """
    if type(dtype) is DataType:
        return dtype
    elif dtype == np.float32:
        return DataType.FLOAT
    elif dtype == np.float16:
        return DataType.HALF
    elif dtype == np.int8:
        return DataType.INT8
    else:
        raise ValueError("unknown datatype")
