import tempfile
import uuid
from pathlib import Path
from typing import Dict, Optional, Union
from urllib.parse import urlparse

import fsspec
from dask.array.core import _check_regular_chunks
from yarl import URL

PathType = Union[str, Path]


def temporary_directory(
    *,
    suffix: Optional[str] = None,
    prefix: Optional[str] = None,
    dir: Optional[PathType] = None,
    storage_options: Optional[Dict[str, str]] = None,
) -> str:
    """Create a temporary directory in a fsspec filesystem.
    Parameters
    ----------
    suffix : Optional[str], optional
        If not None, the name of the temporary directory will end with that suffix.
    prefix : Optional[str], optional
        If not None, the name of the temporary directory will start with that prefix.
    dir : Optional[PathType], optional
        If not None, the temporary directory will be created in that directory, otherwise
        the local filesystem directory returned by `tempfile.gettempdir()` will be used.
        The directory may be specified as any fsspec URL.
    storage_options : Optional[Dict[str, str]], optional
        Any additional parameters for the storage backend (see `fsspec.open`).

    Returns
    -------
    str
        The fsspec URL to the created directory.
    """

    # Fill in defaults
    suffix = suffix or ""
    prefix = prefix or ""
    dir = dir or tempfile.gettempdir()
    storage_options = storage_options or {}

    # Find the filesystem by looking at the URL scheme (protocol), empty means local filesystem
    protocol = urlparse(str(dir)).scheme
    fs = fsspec.filesystem(protocol, **storage_options)

    # Construct a random directory name
    tempdir = build_url(dir, prefix + str(uuid.uuid4()) + suffix)
    fs.mkdir(tempdir)
    return tempdir


def build_url(dir_url: PathType, child_path: str) -> str:
    """Combine a URL for a directory with a child path"""
    url = URL(str(dir_url))
    # the division (/) operator discards query and fragment, so add them back
    return str((url / child_path).with_query(url.query).with_fragment(url.fragment))


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
