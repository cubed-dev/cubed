from importlib.metadata import version as _version

try:
    __version__ = _version("cubed")
except Exception:  # pragma: no cover
    __version__ = "unknown"

from donfig import Config

config = Config(
    "cubed",
    # default spec is local temp dir and a reasonable amount of memory (2GB, of which 100MB is reserved)
    defaults=[{"spec": {"allowed_mem": "2GB", "reserved_mem": "100MB"}}],
)

from .array.nan_functions import nanmean, nansum
from .array.overlap import map_overlap
from .array.pad import pad
from .core.array import compute, measure_reserved_mem, visualize
from .core.gufunc import apply_gufunc
from .core.ops import from_array, from_zarr, map_blocks, rechunk, store, to_zarr
from .runtime.types import Callback, TaskEndEvent
from .spec import Spec

__all__ = [
    "__version__",
    "Callback",
    "Spec",
    "TaskEndEvent",
    "apply_gufunc",
    "compute",
    "config",
    "from_array",
    "from_zarr",
    "map_blocks",
    "map_overlap",
    "measure_reserved_mem",
    "nanmean",
    "nansum",
    "pad",
    "rechunk",
    "store",
    "to_zarr",
    "visualize",
]

# Array API

__array_api_version__ = "2023.12"

from .array_api.inspection import __array_namespace_info__

__all__ += ["__array_api_version__", "__array_namespace_info__"]


from .array_api.array_object import Array

__all__ += ["Array"]

from .array_api.constants import e, inf, nan, newaxis, pi

__all__ += ["e", "inf", "nan", "newaxis", "pi"]

from .array_api.creation_functions import (
    arange,
    asarray,
    empty,
    empty_like,
    eye,
    full,
    full_like,
    linspace,
    meshgrid,
    ones,
    ones_like,
    tril,
    triu,
    zeros,
    zeros_like,
)

__all__ += [
    "arange",
    "asarray",
    "empty",
    "empty_like",
    "eye",
    "full",
    "full_like",
    "linspace",
    "meshgrid",
    "ones",
    "ones_like",
    "tril",
    "triu",
    "zeros",
    "zeros_like",
]

from .array_api.data_type_functions import (
    astype,
    can_cast,
    finfo,
    iinfo,
    isdtype,
    result_type,
)

__all__ += ["astype", "can_cast", "finfo", "iinfo", "isdtype", "result_type"]

from .array_api.dtypes import (
    bool,
    complex64,
    complex128,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)

__all__ += [
    "bool",
    "complex64",
    "complex128",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]

from .array_api.elementwise_functions import (
    abs,
    acos,
    acosh,
    add,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    bitwise_and,
    bitwise_invert,
    bitwise_left_shift,
    bitwise_or,
    bitwise_right_shift,
    bitwise_xor,
    ceil,
    clip,
    conj,
    copysign,
    cos,
    cosh,
    divide,
    equal,
    exp,
    expm1,
    floor,
    floor_divide,
    greater,
    greater_equal,
    hypot,
    imag,
    isfinite,
    isinf,
    isnan,
    less,
    less_equal,
    log,
    log1p,
    log2,
    log10,
    logaddexp,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    maximum,
    minimum,
    multiply,
    negative,
    not_equal,
    positive,
    pow,
    real,
    remainder,
    round,
    sign,
    signbit,
    sin,
    sinh,
    sqrt,
    square,
    subtract,
    tan,
    tanh,
    trunc,
)

__all__ += [
    "abs",
    "acos",
    "acosh",
    "add",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_invert",
    "bitwise_left_shift",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "ceil",
    "clip",
    "conj",
    "copysign",
    "cos",
    "cosh",
    "divide",
    "equal",
    "exp",
    "expm1",
    "floor",
    "floor_divide",
    "greater",
    "greater_equal",
    "hypot",
    "imag",
    "isfinite",
    "isinf",
    "isnan",
    "less",
    "less_equal",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "multiply",
    "negative",
    "not_equal",
    "positive",
    "pow",
    "real",
    "remainder",
    "round",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "subtract",
    "tan",
    "tanh",
    "trunc",
]

from .array_api.indexing_functions import take

__all__ += ["take"]

from .array_api.linear_algebra_functions import (
    matmul,
    matrix_transpose,
    tensordot,
    vecdot,
)

__all__ += ["matmul", "matrix_transpose", "tensordot", "vecdot"]

from .array_api.manipulation_functions import (
    broadcast_arrays,
    broadcast_to,
    concat,
    expand_dims,
    flip,
    moveaxis,
    permute_dims,
    repeat,
    reshape,
    roll,
    squeeze,
    stack,
    tile,
    unstack,
)

__all__ += [
    "broadcast_arrays",
    "broadcast_to",
    "concat",
    "expand_dims",
    "flip",
    "moveaxis",
    "permute_dims",
    "repeat",
    "reshape",
    "roll",
    "squeeze",
    "stack",
    "tile",
    "unstack",
]

from .array_api.searching_functions import argmax, argmin, searchsorted, where

__all__ += ["argmax", "argmin", "searchsorted", "where"]

from .array_api.statistical_functions import max, mean, min, prod, std, sum, var

__all__ += ["max", "mean", "min", "prod", "std", "sum", "var"]

from .array_api.utility_functions import all, any

__all__ += ["all", "any"]

# extensions

from .array_api import linalg

__all__ += ["linalg"]
