import numpy as np
import pytest

from cubed.utils import chunk_memory, join_path, memory_repr, peak_memory, to_chunksize


def test_chunk_memory():
    assert chunk_memory(int, (3,)) == 24
    assert chunk_memory(np.int64, (3,)) == 24
    assert chunk_memory(np.int32, (3,)) == 12
    assert chunk_memory(np.int32, (3, 5)) == 60
    assert chunk_memory(np.int32, (0,)) == 0


def test_to_chunksize():
    assert to_chunksize(((3, 3, 3, 1),)) == (3,)
    with pytest.raises(ValueError):
        to_chunksize(((3, 2, 3, 3, 1),))


def test_join_path():
    assert join_path("http://host/path", "subpath") == "http://host/path/subpath"
    assert join_path("http://host/path/", "subpath") == "http://host/path/subpath"
    assert (
        join_path("http://host/path?a=b", "subpath") == "http://host/path/subpath?a=b"
    )
    assert (
        join_path("http://host/path/?a=b", "subpath") == "http://host/path/subpath?a=b"
    )
    assert join_path("http://host/path#a", "subpath") == "http://host/path/subpath#a"
    assert join_path("s3://host/path", "subpath") == "s3://host/path/subpath"
    assert join_path("relative_path/path", "subpath") == "relative_path/path/subpath"
    assert join_path("/absolute_path/path", "subpath") == "/absolute_path/path/subpath"
    assert (
        join_path("http://host/a%20path", "subpath") == "http://host/a%20path/subpath"
    )
    assert join_path("http://host/a path", "subpath") == "http://host/a%20path/subpath"


def test_memory_repr():
    assert memory_repr(0) == "0 bytes"
    assert memory_repr(1) == "1 bytes"
    assert memory_repr(999) == "999 bytes"
    assert memory_repr(1_000) == "1.0 KB"
    assert memory_repr(9_999) == "10.0 KB"
    assert memory_repr(1_000_000) == "1.0 MB"
    assert memory_repr(1_000_000_000_000_000) == "1.0 PB"


def test_peak_memory():
    assert peak_memory() > 0
