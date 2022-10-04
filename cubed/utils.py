import collections
import itertools
import platform
import sys
import sysconfig
import traceback
from dataclasses import dataclass
from math import prod
from pathlib import Path
from posixpath import join
from resource import RUSAGE_SELF, getrusage
from typing import Union
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import numpy as np
from dask.array.core import _check_regular_chunks
from dask.utils import cached_cumsum

PathType = Union[str, Path]


def chunk_memory(dtype, chunksize):
    """Calculate the amount of memory in bytes that a single chunk uses."""
    return np.dtype(dtype).itemsize * prod(chunksize)


def get_item(chunks, idx):
    """Convert a chunk index to a tuple of slices."""
    starts = tuple(cached_cumsum(c, initial_zero=True) for c in chunks)
    loc = tuple((start[i], start[i + 1]) for i, start in zip(idx, starts))
    return tuple(slice(*s, None) for s in loc)


def join_path(dir_url: PathType, child_path: str) -> str:
    """Combine a URL for a directory with a child path"""
    parts = urlsplit(str(dir_url))
    new_path = quote(join(unquote(parts.path), child_path))
    split_parts = (parts.scheme, parts.netloc, new_path, parts.query, parts.fragment)
    return urlunsplit(split_parts)


def memory_repr(num):
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
    if num < 1000.0:
        return f"{num} bytes"
    num /= 1000.0
    for x in ["KB", "MB", "GB", "TB", "PB"]:
        if num < 1000.0:
            return f"{num:3.1f} {x}"
        num /= 1000.0


def peak_memory():
    """Return the peak memory usage in bytes."""
    ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    # note that on Linux ru_maxrss is in KiB, while on Mac it is in bytes
    # see https://pythonspeed.com/articles/estimating-memory-usage/#measuring-peak-memory-usage
    return ru_maxrss * 1024 if platform.system() == "Linux" else ru_maxrss


def to_chunksize(chunkset):
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
        raise ValueError("Array must have regular chunks")

    return tuple(c[0] for c in chunkset)


@dataclass
class StackSummary:
    """Like Python's ``FrameSummary``, but with module information."""

    filename: str
    lineno: int
    name: str
    module: str

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
        )
        stack_summaries.append(summary)
    stack_summaries.reverse()

    return stack_summaries
