import numbers

import numpy as np


def validate_axis(axis, ndim):
    """Validate an input to axis= keywords"""
    if isinstance(axis, (tuple, list)):
        return tuple(validate_axis(ax, ndim) for ax in axis)
    if not isinstance(axis, numbers.Integral):
        raise TypeError("Axis value must be an integer, got %s" % axis)
    if axis < -ndim or axis >= ndim:
        raise np.AxisError(
            "Axis %d is out of bounds for array of dimension %d" % (axis, ndim)
        )
    if axis < 0:
        axis += ndim
    return axis
