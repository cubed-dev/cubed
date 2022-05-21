import numpy as np

from cubed.array_api.data_type_functions import result_type
from cubed.core import elementwise_binary_operation, elementwise_unary_operation


def add(x1, x2, /):
    return elementwise_binary_operation(x1, x2, np.add, result_type(x1.dtype, x2.dtype))


def divide(x1, x2, /):
    return elementwise_binary_operation(
        x1, x2, np.divide, result_type(x1.dtype, x2.dtype)
    )


def equal(x1, x2, /):
    return elementwise_binary_operation(x1, x2, np.equal, dtype=np.bool)


def negative(x, /):
    return elementwise_unary_operation(x, np.negative, dtype=x.dtype)
