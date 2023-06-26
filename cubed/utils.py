import collections
import itertools
import platform
import sys
import sysconfig
import traceback
from dataclasses import dataclass
from math import prod
from operator import add
from pathlib import Path
from posixpath import join
from typing import Tuple, Union
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import numpy as np
import tlz as toolz

from cubed.types import T_DType, T_RectangularChunks, T_RegularChunks
from cubed.vendor.dask.array.core import _check_regular_chunks

PathType = Union[str, Path]


def chunk_memory(dtype: T_DType, chunksize: T_RegularChunks) -> int:
    """Calculate the amount of memory in bytes that a single chunk uses."""
    return np.dtype(dtype).itemsize * prod(chunksize)


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

    return tuple(c[0] for c in chunkset)


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
            and hasattr(arr, "data")
            and hasattr(arr, "name")
        ):
            if isinstance(arr.data, Array):
                array_names_to_variable_names[arr.data.name] = arr.name
    return array_names_to_variable_names


def convert_to_bytes(size: Union[int, str]) -> int:
    """
    Converts the input data size to bytes.

    The data size can be expressed as an integer or as a string with different SI prefixes such as '500kB', '2MB', or '1GB'.

    Parameters
    ----------
    size: in or str:
        Size of data. If int it should be >=0. If str it should be of form <value><unit> where unit can be kB, MB, GB, TB etc.

    Returns
    -------
    int: The size in bytes
    """
    units = {"B": 0, "kB": 1, "MB": 2, "GB": 3, "TB": 4, "PB": 5}

    if isinstance(size, int) and size >= 0:
        return size
    elif isinstance(size, str):
        # check if the format is valid
        if size[-1] == "B" and size[:-1].isdigit():
            unit = "B"
            value = size[:-1]
        elif size[-2:] in units and size[:-2].isdigit():
            unit = size[-2:]
            value = size[:-2]
        else:
            raise ValueError(
                f"Invalid value: {size}. Expected a string ending with an SI prefix."
            )

        if unit in units and value.isdigit():
            # convert to bytes
            return int(value) * (1000 ** units[unit])
    raise ValueError(
        f"Invalid value: {size}. Expected a positive integer or a string ending with an SI prefix."
    )
