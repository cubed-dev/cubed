import os

_HANDLED_FUNCTIONS = {}

CUBED_NUMPY_COMPAT = "CUBED_NUMPY_COMPAT" in os.environ


def implements(*numpy_functions):
    """Register a cubed implementation of one or more numpy functions."""

    def decorator(cubed_function):
        for numpy_function in numpy_functions:
            _HANDLED_FUNCTIONS[numpy_function] = cubed_function
        return cubed_function

    return decorator


_UFUNC_MAP = None


def _get_ufunc_func(numpy_ufunc):
    global _UFUNC_MAP
    if _UFUNC_MAP is None:
        _UFUNC_MAP = _build_ufunc_map()
    return _UFUNC_MAP.get(numpy_ufunc)


def _build_ufunc_map():
    import numpy as np

    import cubed.array_api.elementwise_functions as ef

    return {
        # unary
        np.absolute: ef.abs,
        np.arccos: ef.acos,
        np.arccosh: ef.acosh,
        np.arcsin: ef.asin,
        np.arcsinh: ef.asinh,
        np.arctan: ef.atan,
        np.arctanh: ef.atanh,
        np.ceil: ef.ceil,
        np.conjugate: ef.conj,
        np.cos: ef.cos,
        np.cosh: ef.cosh,
        np.exp: ef.exp,
        np.expm1: ef.expm1,
        np.floor: ef.floor,
        np.isfinite: ef.isfinite,
        np.isinf: ef.isinf,
        np.isnan: ef.isnan,
        np.log: ef.log,
        np.log1p: ef.log1p,
        np.log2: ef.log2,
        np.log10: ef.log10,
        np.logical_not: ef.logical_not,
        np.negative: ef.negative,
        np.positive: ef.positive,
        np.sign: ef.sign,
        np.signbit: ef.signbit,
        np.sin: ef.sin,
        np.sinh: ef.sinh,
        np.sqrt: ef.sqrt,
        np.square: ef.square,
        np.tan: ef.tan,
        np.tanh: ef.tanh,
        np.trunc: ef.trunc,
        # binary
        np.add: ef.add,
        np.arctan2: ef.atan2,
        np.bitwise_and: ef.bitwise_and,
        np.bitwise_or: ef.bitwise_or,
        np.bitwise_xor: ef.bitwise_xor,
        np.copysign: ef.copysign,
        np.divide: ef.divide,
        np.true_divide: ef.divide,
        np.equal: ef.equal,
        np.floor_divide: ef.floor_divide,
        np.greater: ef.greater,
        np.greater_equal: ef.greater_equal,
        np.hypot: ef.hypot,
        np.left_shift: ef.bitwise_left_shift,
        np.less: ef.less,
        np.less_equal: ef.less_equal,
        np.logaddexp: ef.logaddexp,
        np.logical_and: ef.logical_and,
        np.logical_or: ef.logical_or,
        np.logical_xor: ef.logical_xor,
        np.maximum: ef.maximum,
        np.minimum: ef.minimum,
        np.multiply: ef.multiply,
        np.nextafter: ef.nextafter,
        np.not_equal: ef.not_equal,
        np.power: ef.pow,
        np.remainder: ef.remainder,
        np.right_shift: ef.bitwise_right_shift,
        np.subtract: ef.subtract,
    }
