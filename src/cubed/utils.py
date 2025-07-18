import collections
import itertools
import numbers
import platform
import sys
import sysconfig
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from itertools import islice
from math import prod
from operator import add, mul
from pathlib import Path
from posixpath import join
from typing import Dict, Tuple, Union, cast
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import numpy as np
import tlz as toolz
from toolz import reduce

from cubed.backend_array_api import namespace as nxp
from cubed.types import T_DType, T_RectangularChunks, T_RegularChunks, T_Shape
from cubed.vendor.dask.array.core import _check_regular_chunks, normalize_chunks

PathType = Union[str, Path]


def array_memory(dtype: T_DType, shape: T_Shape) -> int:
    """Calculate the amount of memory in bytes that an array uses."""
    return np.dtype(dtype).itemsize * prod(shape)


def chunk_memory(arr) -> int:
    """Calculate the amount of memory in bytes that a single chunk uses."""
    if hasattr(arr, "chunkmem"):
        return arr.chunkmem
    return array_memory(
        arr.dtype,
        to_chunksize(normalize_chunks(arr.chunks, shape=arr.shape, dtype=arr.dtype)),
    )


def array_size(shape: T_Shape) -> int:
    """Number of elements in an array."""
    return reduce(mul, shape, 1)


def offset_to_block_id(offset: int, numblocks: Tuple[int, ...]) -> Tuple[int, ...]:
    """Convert an index offset to a block ID (chunk coordinates)."""
    return tuple(int(i) for i in np.unravel_index(offset, numblocks))


def block_id_to_offset(block_id: Tuple[int, ...], numblocks: Tuple[int, ...]) -> int:
    """Convert a block ID (chunk coordinates) to an index offset."""
    return int(np.ravel_multi_index(block_id, numblocks))


def get_item(chunks: T_RectangularChunks, idx: Tuple[int, ...]) -> Tuple[slice, ...]:
    """Convert a chunk index to a tuple of slices."""
    # could use Dask's cached_cumsum here if it improves performance
    starts = tuple(_cumsum(c, initial_zero=True) for c in chunks)
    loc = tuple((start[i], start[i + 1]) for i, start in zip(idx, starts))
    return tuple(slice(*s, None) for s in loc)


def _cumsum(seq, initial_zero=False):
    if initial_zero:
        return tuple(toolz.accumulate(add, seq, 0))
    else:
        return tuple(toolz.accumulate(add, seq))


def join_path(dir_url: PathType, child_path: str) -> str:
    """Combine a URL for a directory with a child path"""
    parts = urlsplit(str(dir_url))
    new_path = quote(join(unquote(parts.path), child_path))
    split_parts = (parts.scheme, parts.netloc, new_path, parts.query, parts.fragment)
    return urlunsplit(split_parts)


def is_local_path(path: PathType):
    """Determine if a path string is for the local filesystem."""
    return urlsplit(str(path)).scheme in ("", "file")


def is_cloud_storage_path(path: PathType):
    """Determine if a path string is for cloud storage."""
    return urlsplit(str(path)).scheme in ("gs", "s3")


def memory_repr(num: int) -> str:
    """Convert bytes to a human-readable string in decimal form.
    1 KB is 1,000 bytes, 1 MB is 1,000,000 bytes, and so on.

    Parameters
    ----------
    num: int
        Number of bytes

    Returns
    -------
    str
    """
    if num < 0:
        raise ValueError(f"Invalid value: {num}. Expected a positive integer.")
    if num < 1000.0:
        return f"{num} bytes"
    val = num / 1000.0
    for x in ["KB", "MB", "GB", "TB", "PB"]:
        if val < 1000.0:
            return f"{val:3.1f} {x}"
        val /= 1000.0
    # fall back to scientific notation
    return f"{num:.1e} bytes"


def peak_measured_mem() -> int:
    """Measures the peak memory usage in bytes.

    Note: this function currently doesn't work on Windows.
    """

    if platform.system() == "Windows":
        raise NotImplementedError("`peak_measured_mem` is not implemented on Windows")

    from resource import RUSAGE_SELF, getrusage

    ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    # note that on Linux ru_maxrss is in KiB, while on Mac it is in bytes
    # see https://pythonspeed.com/articles/estimating-memory-usage/#measuring-peak-memory-usage
    return ru_maxrss * 1024 if platform.system() == "Linux" else ru_maxrss


def to_chunksize(chunkset: T_RectangularChunks) -> T_RegularChunks:
    """Convert a chunkset to a chunk size for Zarr.

    Parameters
    ----------
    chunkset: tuple of tuples of ints
        From the ``.chunks`` attribute of an ``Array``

    Returns
    -------
    Tuple of ints
    """

    if not _check_regular_chunks(chunkset):
        raise ValueError(f"Array must have regular chunks, but found chunks={chunkset}")

    return tuple(max(c[0], 1) for c in chunkset)


def numblocks(chunks: T_RectangularChunks) -> Tuple[int, ...]:
    return tuple(map(len, chunks))


@dataclass
class StackSummary:
    """Like Python's ``FrameSummary``, but with module information."""

    filename: str
    lineno: int
    name: str
    module: str
    array_names_to_variable_names: dict

    def is_cubed(self):
        """Return True if this stack frame is a Cubed call."""
        return self.module.startswith("cubed.") and not self.module.startswith(
            "cubed.tests."
        )

    def is_on_python_lib_path(self):
        """Return True if this stack frame is from a library on Python's library path."""
        python_lib_path = sysconfig.get_path("purelib")

        return self.filename.startswith(python_lib_path)


def extract_stack_summaries(frame, limit=None):
    """Like Python's ``StackSummary.extract``, but returns module information."""
    frame_gen = traceback.walk_stack(frame)

    # from StackSummary.extract
    if limit is None:
        limit = getattr(sys, "tracebacklimit", None)
        if limit is not None and limit < 0:
            limit = 0
    if limit is not None:
        if limit >= 0:
            frame_gen = itertools.islice(frame_gen, limit)
        else:
            frame_gen = collections.deque(frame_gen, maxlen=-limit)
    # end from StackSummary.extract

    stack_summaries = []
    for f, _ in frame_gen:
        module = f.f_globals.get("__name__", "")
        summary = StackSummary(
            filename=f.f_code.co_filename,
            lineno=f.f_lineno,
            name=f.f_code.co_name,
            module=module,
            array_names_to_variable_names=extract_array_names(f),
        )
        stack_summaries.append(summary)
    stack_summaries.reverse()

    return stack_summaries


def extract_array_names(frame):
    """Look for Cubed arrays in local variables to create a mapping from (internally generated) array names to variable names."""

    from cubed import Array

    array_names_to_variable_names = {}
    for var, arr in frame.f_locals.items():
        if isinstance(arr, Array):
            array_names_to_variable_names[arr.name] = var
        elif (
            type(arr).__module__.split(".")[0] == "xarray"
            and arr.__class__.__name__ == "DataArray"
        ):
            if isinstance(arr.data, Array):
                array_names_to_variable_names[arr.data.name] = arr.name
    return array_names_to_variable_names


def convert_to_bytes(size: Union[int, float, str]) -> int:
    """
    Converts the input data size to bytes.

    The data size can be expressed as an integer or as a string with different SI prefixes such as '500kB', '2MB', or '1GB'.

    Parameters
    ----------
    size: int, float, or str:
        Size of data. If numeric it should represent an integer >=0. If str it should be of form <value><unit> where unit can be B, kB, MB, GB, TB etc.

    Returns
    -------
    int: The size in bytes
    """
    units: Dict[str, int] = {"kB": 1, "MB": 2, "GB": 3, "TB": 4, "PB": 5}

    def is_numeric_str(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    if isinstance(size, str):
        size = size.replace(" ", "")

        # check if the format of the string is valid
        if is_numeric_str(size):
            unit_factor = 1.0
            value = size
        elif size[-1] == "B" and is_numeric_str(size[:-1]):
            unit_factor = 1.0
            value = size[:-1]
        elif size[-2:] in units and is_numeric_str(size[:-2]):
            unit = size[-2:]
            unit_factor = 1000 ** units[unit]
            value = size[:-2]
        else:
            raise ValueError(
                f"Invalid value: {size}. Expected the string to be a numeric value ending with an SI prefix."
            )

        # convert to float number of bytes
        size = float(value) * unit_factor

    if isinstance(size, float):
        if size.is_integer():
            size = int(size)
        else:
            raise ValueError(
                f"Invalid value: {size}. Can't have a non-integer number of bytes"
            )

    if size >= 0:
        return size
    else:
        raise ValueError(f"Invalid value: {size}. Must be a positive value")


# Based on more_itertools
def split_into(iterable, sizes):
    """Yield a list of sequential items from *iterable* of length 'n' for each
    integer 'n' in *sizes*."""
    it = iter(iterable)
    for size in sizes:
        yield list(islice(it, size))


def map_nested(func, seq):
    """Apply a function inside nested lists or iterators, while preserving
    the nesting, and the collection or iterator type.

    Examples
    --------

    >>> from cubed.utils import map_nested
    >>> inc = lambda x: x + 1
    >>> map_nested(inc, [[1, 2], [3, 4]])
    [[2, 3], [4, 5]]

    >>> it = map_nested(inc, iter([1, 2]))
    >>> next(it)
    2
    >>> next(it)
    3
    """
    if isinstance(seq, list):
        return [map_nested(func, item) for item in seq]
    elif isinstance(seq, Iterator):
        return map(lambda item: map_nested(func, item), seq)
    else:
        return func(seq)


def _broadcast_trick_inner(func, shape, *args, **kwargs):
    # cupy-specific hack. numpy is happy with hardcoded shape=().
    null_shape = () if shape == () else 1

    return nxp.broadcast_to(func(*args, shape=null_shape, **kwargs), shape)


def broadcast_trick(func):
    """Apply Dask's broadcast trick to array API functions that produce arrays
    containing a single value to save space in memory.

    Note that this should only be used for arrays that never mutated.
    """
    inner = partial(_broadcast_trick_inner, func)
    inner.__doc__ = func.__doc__
    inner.__name__ = func.__name__
    return inner


def normalize_shape(shape: Union[int, Tuple[int, ...], None]) -> Tuple[int, ...]:
    """Normalize a `shape` argument to a tuple of ints."""

    if shape is None:
        raise TypeError("shape is None")

    if isinstance(shape, numbers.Integral):
        shape = (int(shape),)

    shape = cast(Tuple[int, ...], shape)
    shape = tuple(int(s) for s in shape)
    return shape
