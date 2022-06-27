import numpy as np

from cubed.array_api.data_type_functions import result_type
from cubed.core.ops import elemwise


def where(condition, x1, x2, /):
    dtype = result_type(x1, x2)
    return elemwise(np.where, condition, x1, x2, dtype=dtype)
