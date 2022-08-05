import platform
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
    parts = (parts.scheme, parts.netloc, new_path, parts.query, parts.fragment)
    return urlunsplit(parts)


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
