import operator

import numpy as np

from cubed.array_api.creation_functions import asarray
from cubed.array_api.data_type_functions import result_type
from cubed.array_api.dtypes import (
    _boolean_dtypes,
    _dtype_categories,
    _floating_dtypes,
    _integer_or_boolean_dtypes,
    _numeric_dtypes,
)
from cubed.array_api.linear_algebra_functions import matmul
from cubed.core.array import CoreArray
from cubed.core.ops import elemwise


class Array(CoreArray):
    def __init__(self, name, zarray, spec, plan):
        super().__init__(name, zarray, spec, plan)

    def __array__(self, dtype=None):
        x = self.compute()
        if dtype and x.dtype != dtype:
            x = x.astype(dtype)
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return x

    # Attributes

    @property
    def device(self):
        return "cpu"

    @property
    def mT(self):
        from cubed.array_api.linear_algebra_functions import matrix_transpose

        return matrix_transpose(self)

    @property
    def T(self):
        if self.ndim != 2:
            raise ValueError("x.T requires x to have 2 dimensions.")
        from cubed.array_api.linear_algebra_functions import matrix_transpose

        return matrix_transpose(self)

    # Arithmetic Operators

    def __neg__(self, /):
        if self.dtype not in _numeric_dtypes:
            raise TypeError("Only numeric dtypes are allowed in __neg__")
        return elemwise(np.negative, self, dtype=self.dtype)

    def __pos__(self, /):
        if self.dtype not in _numeric_dtypes:
            raise TypeError("Only numeric dtypes are allowed in __pos__")
        return elemwise(np.positive, self, dtype=self.dtype)

    def __add__(self, other, /):
        other = self._check_allowed_dtypes(other, "numeric", "__add__")
        if other is NotImplemented:
            return other
        return elemwise(np.add, self, other, dtype=result_type(self, other))

    def __sub__(self, other, /):
        other = self._check_allowed_dtypes(other, "numeric", "__sub__")
        if other is NotImplemented:
            return other
        return elemwise(np.subtract, self, other, dtype=result_type(self, other))

    def __mul__(self, other, /):
        other = self._check_allowed_dtypes(other, "numeric", "__mul__")
        if other is NotImplemented:
            return other
        return elemwise(np.multiply, self, other, dtype=result_type(self, other))

    def __truediv__(self, other, /):
        other = self._check_allowed_dtypes(other, "floating-point", "__truediv__")
        if other is NotImplemented:
            return other
        return elemwise(np.divide, self, other, dtype=result_type(self, other))

    def __floordiv__(self, other, /):
        other = self._check_allowed_dtypes(other, "numeric", "__floordiv__")
        if other is NotImplemented:
            return other
        return elemwise(np.floor_divide, self, other, dtype=result_type(self, other))

    def __mod__(self, other, /):
        other = self._check_allowed_dtypes(other, "numeric", "__mod__")
        if other is NotImplemented:
            return other
        return elemwise(np.remainder, self, other, dtype=result_type(self, other))

    def __pow__(self, other, /):
        other = self._check_allowed_dtypes(other, "numeric", "__pow__")
        if other is NotImplemented:
            return other
        return elemwise(np.power, self, other, dtype=result_type(self, other))

    # Array Operators

    def __matmul__(self, other, /):
        other = self._check_allowed_dtypes(other, "numeric", "__matmul__")
        if other is NotImplemented:
            return other
        return matmul(self, other)

    # Bitwise Operators

    def __invert__(self, /):
        if self.dtype not in _integer_or_boolean_dtypes:
            raise TypeError("Only integer or boolean dtypes are allowed in __invert__")
        return elemwise(np.invert, self, dtype=self.dtype)

    def __and__(self, other, /):
        other = self._check_allowed_dtypes(other, "integer or boolean", "__and__")
        if other is NotImplemented:
            return other
        return elemwise(np.bitwise_and, self, other, dtype=result_type(self, other))

    def __or__(self, other, /):
        other = self._check_allowed_dtypes(other, "integer or boolean", "__or__")
        if other is NotImplemented:
            return other
        return elemwise(np.bitwise_or, self, other, dtype=result_type(self, other))

    def __xor__(self, other, /):
        other = self._check_allowed_dtypes(other, "integer or boolean", "__xor__")
        if other is NotImplemented:
            return other
        return elemwise(np.bitwise_xor, self, other, dtype=result_type(self, other))

    def __lshift__(self, other, /):
        other = self._check_allowed_dtypes(other, "integer", "__lshift__")
        if other is NotImplemented:
            return other
        return elemwise(np.left_shift, self, other, dtype=result_type(self, other))

    def __rshift__(self, other, /):
        other = self._check_allowed_dtypes(other, "integer", "__rshift__")
        if other is NotImplemented:
            return other
        return elemwise(np.right_shift, self, other, dtype=result_type(self, other))

    # Comparison Operators

    def __eq__(self, other, /):
        other = self._check_allowed_dtypes(other, "all", "__eq__")
        if other is NotImplemented:
            return other
        return elemwise(np.equal, self, other, dtype=np.bool_)

    def __ge__(self, other, /):
        other = self._check_allowed_dtypes(other, "all", "__ge__")
        if other is NotImplemented:
            return other
        return elemwise(np.greater_equal, self, other, dtype=np.bool_)

    def __gt__(self, other, /):
        other = self._check_allowed_dtypes(other, "all", "__gt__")
        if other is NotImplemented:
            return other
        return elemwise(np.greater, self, other, dtype=np.bool_)

    def __le__(self, other, /):
        other = self._check_allowed_dtypes(other, "all", "__le__")
        if other is NotImplemented:
            return other
        return elemwise(np.less_equal, self, other, dtype=np.bool_)

    def __lt__(self, other, /):
        other = self._check_allowed_dtypes(other, "all", "__lt__")
        if other is NotImplemented:
            return other
        return elemwise(np.less, self, other, dtype=np.bool_)

    def __ne__(self, other, /):
        other = self._check_allowed_dtypes(other, "all", "__ne__")
        if other is NotImplemented:
            return other
        return elemwise(np.not_equal, self, other, dtype=np.bool_)

    # Reflected Operators

    # (Reflected) Arithmetic Operators

    def __radd__(self, other, /):
        other = self._check_allowed_dtypes(other, "numeric", "__radd__")
        if other is NotImplemented:
            return other
        return elemwise(np.add, other, self, dtype=result_type(self, other))

    def __rsub__(self, other, /):
        other = self._check_allowed_dtypes(other, "numeric", "__rsub__")
        if other is NotImplemented:
            return other
        return elemwise(np.subtract, other, self, dtype=result_type(self, other))

    def __rmul__(self, other, /):
        other = self._check_allowed_dtypes(other, "numeric", "__rmul__")
        if other is NotImplemented:
            return other
        return elemwise(np.multiply, other, self, dtype=result_type(self, other))

    def __rtruediv__(self, other, /):
        other = self._check_allowed_dtypes(other, "floating-point", "__rtruediv__")
        if other is NotImplemented:
            return other
        return elemwise(np.divide, other, self, dtype=result_type(self, other))

    def __rfloordiv__(self, other, /):
        other = self._check_allowed_dtypes(other, "numeric", "__rfloordiv__")
        if other is NotImplemented:
            return other
        return elemwise(np.floor_divide, other, self, dtype=result_type(self, other))

    def __rmod__(self, other, /):
        other = self._check_allowed_dtypes(other, "numeric", "__rmod__")
        if other is NotImplemented:
            return other
        return elemwise(np.remainder, other, self, dtype=result_type(self, other))

    def __rpow__(self, other, /):
        other = self._check_allowed_dtypes(other, "numeric", "__rpow__")
        if other is NotImplemented:
            return other
        return elemwise(np.power, other, self, dtype=result_type(self, other))

    # (Reflected) Array Operators

    def __rmatmul__(self, other, /):
        other = self._check_allowed_dtypes(other, "numeric", "__rmatmul__")
        if other is NotImplemented:
            return other
        return matmul(other, self)

    # (Reflected) Bitwise Operators

    def __rand__(self, other, /):
        other = self._check_allowed_dtypes(other, "integer or boolean", "__rand__")
        if other is NotImplemented:
            return other
        return elemwise(np.bitwise_and, other, self, dtype=result_type(self, other))

    def __ror__(self, other, /):
        other = self._check_allowed_dtypes(other, "integer or boolean", "__ror__")
        if other is NotImplemented:
            return other
        return elemwise(np.bitwise_or, other, self, dtype=result_type(self, other))

    def __rxor__(self, other, /):
        other = self._check_allowed_dtypes(other, "integer or boolean", "__rxor__")
        if other is NotImplemented:
            return other
        return elemwise(np.bitwise_xor, other, self, dtype=result_type(self, other))

    def __rlshift__(self, other, /):
        other = self._check_allowed_dtypes(other, "integer", "__rlshift__")
        if other is NotImplemented:
            return other
        return elemwise(np.left_shift, other, self, dtype=result_type(self, other))

    def __rrshift__(self, other, /):
        other = self._check_allowed_dtypes(other, "integer", "__rrshift__")
        if other is NotImplemented:
            return other
        return elemwise(np.right_shift, other, self, dtype=result_type(self, other))

    # Methods

    def __abs__(self, /):
        if self.dtype not in _numeric_dtypes:
            raise TypeError("Only numeric dtypes are allowed in __abs__")
        return elemwise(np.abs, self, dtype=self.dtype)

    def __array_namespace__(self, /, *, api_version=None):
        if api_version is not None and not api_version.startswith("2021."):
            raise ValueError(f"Unrecognized array API version: {api_version!r}")
        import cubed.array_api as array_api

        return array_api

    def __bool__(self, /):
        if self.ndim != 0:
            raise TypeError("bool is only allowed on arrays with 0 dimensions")
        return bool(self.compute())

    def __float__(self, /):
        if self.ndim != 0:
            raise TypeError("float is only allowed on arrays with 0 dimensions")
        return float(self.compute())

    def __index__(self, /):
        if self.ndim != 0:
            raise TypeError("index is only allowed on arrays with 0 dimensions")
        return operator.index(self.compute())

    def __int__(self, /):
        if self.ndim != 0:
            raise TypeError("int is only allowed on arrays with 0 dimensions")
        return int(self.compute())

    # Utility methods

    def _check_allowed_dtypes(self, other, dtype_category, op):
        if self.dtype not in _dtype_categories[dtype_category]:
            raise TypeError(f"Only {dtype_category} dtypes are allowed in {op}")
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        elif isinstance(other, CoreArray):
            if other.dtype not in _dtype_categories[dtype_category]:
                raise TypeError(f"Only {dtype_category} dtypes are allowed in {op}")
        else:
            return NotImplemented

        # TODO: more from numpy.array_api

        return other

    def _promote_scalar(self, scalar):
        if isinstance(scalar, bool):
            if self.dtype not in _boolean_dtypes:
                raise TypeError(
                    "Python bool scalars can only be promoted with bool arrays"
                )
        elif isinstance(scalar, int):
            if self.dtype in _boolean_dtypes:
                raise TypeError(
                    "Python int scalars cannot be promoted with bool arrays"
                )
        elif isinstance(scalar, float):
            if self.dtype not in _floating_dtypes:
                raise TypeError(
                    "Python float scalars can only be promoted with floating-point arrays."
                )
        else:
            raise TypeError("'scalar' must be a Python scalar")
        return asarray(scalar, dtype=self.dtype, spec=self.spec)
