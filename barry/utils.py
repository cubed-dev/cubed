from pathlib import Path
from posixpath import join
from typing import Union
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from dask.array.core import _check_regular_chunks

PathType = Union[str, Path]


def join_path(dir_url: PathType, child_path: str) -> str:
    """Combine a URL for a directory with a child path"""
    parts = urlsplit(str(dir_url))
    new_path = quote(join(unquote(parts.path), child_path))
    parts = (parts.scheme, parts.netloc, new_path, parts.query, parts.fragment)
    return urlunsplit(parts)


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
