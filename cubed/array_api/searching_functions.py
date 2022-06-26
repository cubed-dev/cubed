import numpy as np

from cubed.core.ops import elemwise


def where(condition, x1, x2, /):
    return elemwise(np.where, condition, x1, x2)
