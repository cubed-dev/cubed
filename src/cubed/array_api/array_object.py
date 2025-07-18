import math
import operator

import numpy as np

from cubed.array_api.creation_functions import asarray
from cubed.array_api.data_type_functions import result_type
from cubed.array_api.dtypes import (
    _boolean_dtypes,
    _complex_floating_dtypes,
    _dtype_categories,
    _floating_dtypes,
    _integer_dtypes,
    _integer_or_boolean_dtypes,
    _numeric_dtypes,
    complex64,
    complex128,
    float32,
    float64,
)
from cubed.array_api.linear_algebra_functions import matmul
from cubed.backend_array_api import namespace as nxp
from cubed.core.array import CoreArray
from cubed.core.ops import elemwise
from cubed.utils import memory_repr
from cubed.vendor.dask.widgets import get_template

ARRAY_SVG_SIZE = (
    120  # cubed doesn't have a config module like dask does so hard-code this for now
)


class Array(CoreArray):
    """Chunked array backed by Zarr storage that conforms to the Python Array API standard."""

    def __init__(self, name, zarray, spec, plan):
        super().__init__(name, zarray, spec, plan)

    def __array__(self, dtype=None) -> np.ndarray:
        x = self.compute()
        if dtype and x.dtype != dtype:
            x = x.astype(dtype)
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return x

    def __repr__(self):
        return f"cubed.Array<{self.name}, shape={self.shape}, dtype={self.dtype}, chunks={self.chunks}>"

    def _repr_html_(self):
        try:
            grid = self.to_svg(size=ARRAY_SVG_SIZE)
        except NotImplementedError:
            grid = ""

        if not math.isnan(self.nbytes):
            nbytes = memory_repr(self.nbytes)
            cbytes = memory_repr(math.prod(self.chunksize) * self.dtype.itemsize)
        else:
            nbytes = "unknown"
            cbytes = "unknown"

        return get_template("array.html.j2").render(
            array=self,
            grid=grid,
            nbytes=nbytes,
            cbytes=cbytes,
            arrs_in_plan=f"{self.plan._finalize().num_arrays()} arrays in Plan",
            arrtype="np.ndarray",
        )

    def to_svg(self, size=500):
        """Convert chunks from Cubed Array into an SVG Image

        Parameters
        ----------
        chunks: tuple
        size: int
            Rough size of the image

        Examples
        --------
        >>> x.to_svg(size=500)  # doctest: +SKIP

        Returns
        -------
        text: An svg string depicting the array as a grid of chunks
        """
        from cubed.vendor.dask.array.svg import svg

        return svg(self.chunks, size=size)

    def _repr_inline_(self, max_width):
        """
        Format to a single line with at most max_width characters. Used by xarray.
        """
        return f"cubed.Array<chunksize={self.chunksize}>"

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
        return elemwise(nxp.negative, self, dtype=self.dtype)

    def __pos__(self, /):
        if self.dtype not in _numeric_dtypes:
            raise TypeError("Only numeric dtypes are allowed in __pos__")
        return elemwise(nxp.positive, self, dtype=self.dtype)

    def __add__(self, other, /):
        other = self._check_allowed_dtypes(other, "numeric", "__add__")
        if other is NotImplemented:
            return other
        return elemwise(nxp.add, self, other, dtype=result_type(self, other))

    def __sub__(self, other, /):
        other = self._check_allowed_dtypes(other, "numeric", "__sub__")
        if other is NotImplemented:
            return other
        return elemwise(nxp.subtract, self, other, dtype=result_type(self, other))

    def __mul__(self, other, /):
        other = self._check_allowed_dtypes(other, "numeric", "__mul__")
        if other is NotImplemented:
            return other
        return elemwise(nxp.multiply, self, other, dtype=result_type(self, other))

    def __truediv__(self, other, /):
        other = self._check_allowed_dtypes(other, "floating-point", "__truediv__")
        if other is NotImplemented:
            return other
        return elemwise(nxp.divide, self, other, dtype=result_type(self, other))

    def __floordiv__(self, other, /):
        other = self._check_allowed_dtypes(other, "real numeric", "__floordiv__")
        if other is NotImplemented:
            return other
        return elemwise(nxp.floor_divide, self, other, dtype=result_type(self, other))

    def __mod__(self, other, /):
        other = self._check_allowed_dtypes(other, "real numeric", "__mod__")
        if other is NotImplemented:
            return other
        return elemwise(nxp.remainder, self, other, dtype=result_type(self, other))

    def __pow__(self, other, /):
        other = self._check_allowed_dtypes(other, "numeric", "__pow__")
        if other is NotImplemented:
            return other
        return elemwise(nxp.pow, self, other, dtype=result_type(self, other))

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
        return elemwise(nxp.bitwise_invert, self, dtype=self.dtype)

    def __and__(self, other, /):
        other = self._check_allowed_dtypes(other, "integer or boolean", "__and__")
        if other is NotImplemented:
            return other
        return elemwise(nxp.bitwise_and, self, other, dtype=result_type(self, other))

    def __or__(self, other, /):
        other = self._check_allowed_dtypes(other, "integer or boolean", "__or__")
        if other is NotImplemented:
            return other
        return elemwise(nxp.bitwise_or, self, other, dtype=result_type(self, other))

    def __xor__(self, other, /):
        other = self._check_allowed_dtypes(other, "integer or boolean", "__xor__")
        if other is NotImplemented:
            return other
        return elemwise(nxp.bitwise_xor, self, other, dtype=result_type(self, other))

    def __lshift__(self, other, /):
        other = self._check_allowed_dtypes(other, "integer", "__lshift__")
        if other is NotImplemented:
            return other
        return elemwise(
            nxp.bitwise_left_shift, self, other, dtype=result_type(self, other)
        )

    def __rshift__(self, other, /):
        other = self._check_allowed_dtypes(other, "integer", "__rshift__")
        if other is NotImplemented:
            return other
        return elemwise(
            nxp.bitwise_right_shift, self, other, dtype=result_type(self, other)
        )

    # Comparison Operators

    def __eq__(self, other, /):
        other = self._check_allowed_dtypes(other, "all", "__eq__")
        if other is NotImplemented:
            return other
        return elemwise(nxp.equal, self, other, dtype=nxp.bool)

    def __ge__(self, other, /):
        other = self._check_allowed_dtypes(other, "all", "__ge__")
        if other is NotImplemented:
            return other
        return elemwise(nxp.greater_equal, self, other, dtype=nxp.bool)

    def __gt__(self, other, /):
        other = self._check_allowed_dtypes(other, "all", "__gt__")
        if other is NotImplemented:
            return other
        return elemwise(nxp.greater, self, other, dtype=nxp.bool)

    def __le__(self, other, /):
        other = self._check_allowed_dtypes(other, "all", "__le__")
        if other is NotImplemented:
            return other
        return elemwise(nxp.less_equal, self, other, dtype=nxp.bool)

    def __lt__(self, other, /):
        other = self._check_allowed_dtypes(other, "all", "__lt__")
        if other is NotImplemented:
            return other
        return elemwise(nxp.less, self, other, dtype=nxp.bool)

    def __ne__(self, other, /):
        other = self._check_allowed_dtypes(other, "all", "__ne__")
        if other is NotImplemented:
            return other
        return elemwise(nxp.not_equal, self, other, dtype=nxp.bool)

    # Reflected Operators

    # (Reflected) Arithmetic Operators

    def __radd__(self, other, /):
        other = self._check_allowed_dtypes(other, "numeric", "__radd__")
        if other is NotImplemented:
            return other
        return elemwise(nxp.add, other, self, dtype=result_type(self, other))

    def __rsub__(self, other, /):
        other = self._check_allowed_dtypes(other, "numeric", "__rsub__")
        if other is NotImplemented:
            return other
        return elemwise(nxp.subtract, other, self, dtype=result_type(self, other))

    def __rmul__(self, other, /):
        other = self._check_allowed_dtypes(other, "numeric", "__rmul__")
        if other is NotImplemented:
            return other
        return elemwise(nxp.multiply, other, self, dtype=result_type(self, other))

    def __rtruediv__(self, other, /):
        other = self._check_allowed_dtypes(other, "floating-point", "__rtruediv__")
        if other is NotImplemented:
            return other
        return elemwise(nxp.divide, other, self, dtype=result_type(self, other))

    def __rfloordiv__(self, other, /):
        other = self._check_allowed_dtypes(other, "numeric", "__rfloordiv__")
        if other is NotImplemented:
            return other
        return elemwise(nxp.floor_divide, other, self, dtype=result_type(self, other))

    def __rmod__(self, other, /):
        other = self._check_allowed_dtypes(other, "numeric", "__rmod__")
        if other is NotImplemented:
            return other
        return elemwise(nxp.remainder, other, self, dtype=result_type(self, other))

    def __rpow__(self, other, /):
        other = self._check_allowed_dtypes(other, "numeric", "__rpow__")
        if other is NotImplemented:
            return other
        return elemwise(nxp.pow, other, self, dtype=result_type(self, other))

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
        return elemwise(nxp.bitwise_and, other, self, dtype=result_type(self, other))

    def __ror__(self, other, /):
        other = self._check_allowed_dtypes(other, "integer or boolean", "__ror__")
        if other is NotImplemented:
            return other
        return elemwise(nxp.bitwise_or, other, self, dtype=result_type(self, other))

    def __rxor__(self, other, /):
        other = self._check_allowed_dtypes(other, "integer or boolean", "__rxor__")
        if other is NotImplemented:
            return other
        return elemwise(nxp.bitwise_xor, other, self, dtype=result_type(self, other))

    def __rlshift__(self, other, /):
        other = self._check_allowed_dtypes(other, "integer", "__rlshift__")
        if other is NotImplemented:
            return other
        return elemwise(
            nxp.bitwise_left_shift, other, self, dtype=result_type(self, other)
        )

    def __rrshift__(self, other, /):
        other = self._check_allowed_dtypes(other, "integer", "__rrshift__")
        if other is NotImplemented:
            return other
        return elemwise(
            nxp.bitwise_right_shift, other, self, dtype=result_type(self, other)
        )

    # Methods

    def __abs__(self, /):
        if self.dtype not in _numeric_dtypes:
            raise TypeError("Only numeric dtypes are allowed in __abs__")
        if self.dtype == complex64:
            dtype = float32
        elif self.dtype == complex128:
            dtype = float64
        else:
            dtype = self.dtype
        return elemwise(nxp.abs, self, dtype=dtype)

    def __array_namespace__(self, /, *, api_version=None):
        if api_version is not None and api_version not in (
            "2021.12",
            "2022.12",
            "2023.12",
        ):
            raise ValueError(f"Unrecognized array API version: {api_version!r}")
        import cubed

        return cubed

    def __bool__(self, /):
        if self.ndim != 0:
            raise TypeError("bool is only allowed on arrays with 0 dimensions")
        return bool(self.compute())

    def __complex__(self, /):
        if self.ndim != 0:
            raise TypeError("complex is only allowed on arrays with 0 dimensions")
        return complex(self.compute())

    def __float__(self, /):
        if self.ndim != 0:
            raise TypeError("float is only allowed on arrays with 0 dimensions")
        if self.dtype in _complex_floating_dtypes:
            raise TypeError("float is not allowed on complex floating-point arrays")
        return float(self.compute())

    def __index__(self, /):
        if self.ndim != 0:
            raise TypeError("index is only allowed on arrays with 0 dimensions")
        return operator.index(self.compute())

    def __int__(self, /):
        if self.ndim != 0:
            raise TypeError("int is only allowed on arrays with 0 dimensions")
        if self.dtype in _complex_floating_dtypes:
            raise TypeError("int is not allowed on complex floating-point arrays")
        return int(self.compute())

    # Utility methods

    def _check_allowed_dtypes(self, other, dtype_category, op):
        if (
            dtype_category != "all"
            and self.dtype not in _dtype_categories[dtype_category]
        ):
            raise TypeError(f"Only {dtype_category} dtypes are allowed in {op}")
        if isinstance(other, (int, complex, float, bool)):
            other = self._promote_scalar(other)
        elif isinstance(other, CoreArray):
            if (
                dtype_category != "all"
                and other.dtype not in _dtype_categories[dtype_category]
            ):
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
            if self.dtype in _integer_dtypes:
                info = nxp.iinfo(self.dtype)
                if not (info.min <= scalar <= info.max):
                    raise OverflowError(
                        "Python int scalars must be within the bounds of the dtype for integer arrays"
                    )
            # int + array(floating) is allowed
        elif isinstance(scalar, float):
            if self.dtype not in _floating_dtypes:
                raise TypeError(
                    "Python float scalars can only be promoted with floating-point arrays."
                )
        elif isinstance(scalar, complex):
            if self.dtype not in _complex_floating_dtypes:
                raise TypeError(
                    "Python complex scalars can only be promoted with complex floating-point arrays."
                )
        else:
            raise TypeError("'scalar' must be a Python scalar")
        return asarray(scalar, dtype=self.dtype, spec=self.spec)
