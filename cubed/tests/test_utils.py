import inspect
import itertools
import platform

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from cubed.backend_array_api import namespace as nxp
from cubed.utils import (
    array_memory,
    block_id_to_offset,
    broadcast_trick,
    extract_stack_summaries,
    is_cloud_storage_path,
    is_local_path,
    join_path,
    map_nested,
    memory_repr,
    normalize_shape,
    offset_to_block_id,
    peak_measured_mem,
    split_into,
    to_chunksize,
)


def test_array_memory():
    assert array_memory(np.int64, (3,)) == 24
    assert array_memory(np.int32, (3,)) == 12
    assert array_memory(np.int32, (3, 5)) == 60
    assert array_memory(np.int32, (0,)) == 0


def test_block_id_to_offset():
    numblocks = (5, 3)
    for block_id in itertools.product(*[list(range(n)) for n in numblocks]):
        offset = block_id_to_offset(block_id, numblocks)
        assert offset_to_block_id(offset, numblocks) == block_id

    with pytest.raises(ValueError):
        block_id_to_offset((6, 12), numblocks)

    with pytest.raises(ValueError):
        offset_to_block_id(100, numblocks)


def test_to_chunksize():
    assert to_chunksize(((3, 3, 3, 1),)) == (3,)
    assert to_chunksize(((0,),)) == (1,)  # Zarr doesn't support zero-length chunks
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


def test_is_local_path():
    assert is_local_path("relative_path/path")
    assert is_local_path("/absolute_path/path")
    assert is_local_path("file:relative_path/path")
    assert is_local_path("file://absolute_path/path")
    assert is_local_path("file:///absolute_path/path")
    assert not is_local_path("s3://host/path")
    assert not is_local_path("gs://host/path")


def test_is_cloud_storage_path():
    assert not is_cloud_storage_path("relative_path/path")
    assert not is_cloud_storage_path("/absolute_path/path")
    assert not is_cloud_storage_path("file:relative_path/path")
    assert not is_cloud_storage_path("file://absolute_path/path")
    assert not is_cloud_storage_path("file:///absolute_path/path")
    assert is_cloud_storage_path("s3://host/path")
    assert is_cloud_storage_path("gs://host/path")


def test_memory_repr():
    assert memory_repr(0) == "0 bytes"
    assert memory_repr(1) == "1 bytes"
    assert memory_repr(999) == "999 bytes"
    assert memory_repr(1_000) == "1.0 KB"
    assert memory_repr(9_999) == "10.0 KB"
    assert memory_repr(1_000_000) == "1.0 MB"
    assert memory_repr(1_000_000_000_000_000) == "1.0 PB"
    assert memory_repr(int(1e18)) == "1.0e+18 bytes"
    with pytest.raises(ValueError):
        memory_repr(-1)


@pytest.mark.skipif(platform.system() == "Windows", reason="does not run on windows")
def test_peak_measured_mem():
    assert peak_measured_mem() > 0


def test_extract_stack_summaries():
    frame = inspect.currentframe()
    stack_summaries = extract_stack_summaries(frame)
    assert stack_summaries[-1].name == "test_extract_stack_summaries"
    assert stack_summaries[-1].module == "cubed.tests.test_utils"
    assert not stack_summaries[-1].is_cubed()


def test_split_into():
    assert list(split_into([1, 2, 3, 4, 5, 6], [1, 2, 3])) == [[1], [2, 3], [4, 5, 6]]
    assert list(split_into([1, 2, 3, 4, 5, 6], [2, 3])) == [[1, 2], [3, 4, 5]]
    assert list(split_into([1, 2, 3, 4], [1, 2, 3, 4])) == [[1], [2, 3], [4], []]


def test_map_nested_lists():
    inc = lambda x: x + 1

    assert map_nested(inc, [1, 2]) == [2, 3]
    assert map_nested(inc, [[1, 2]]) == [[2, 3]]
    assert map_nested(inc, [[1, 2], [3, 4]]) == [[2, 3], [4, 5]]


count = 0


def inc(x):
    global count
    count = count + 1
    return x + 1


def test_map_nested_iterators():
    # same tests as test_map_nested_lists, but use a counter to check that iterators are advanced at correct points
    global count

    out = map_nested(inc, iter([1, 2]))
    assert isinstance(out, map)
    assert count == 0
    assert next(out) == 2
    assert count == 1
    assert next(out) == 3
    assert count == 2

    # reset count
    count = 0

    out = map_nested(inc, [iter([1, 2])])
    assert isinstance(out, list)
    assert count == 0
    assert len(out) == 1
    out = out[0]
    assert isinstance(out, map)
    assert count == 0
    assert next(out) == 2
    assert count == 1
    assert next(out) == 3
    assert count == 2

    # reset count
    count = 0

    out = map_nested(inc, [iter([1, 2]), iter([3, 4])])
    assert isinstance(out, list)
    assert count == 0
    assert len(out) == 2
    out0 = out[0]
    assert isinstance(out0, map)
    assert count == 0
    assert next(out0) == 2
    assert count == 1
    assert next(out0) == 3
    assert count == 2
    out1 = out[1]
    assert isinstance(out1, map)
    assert count == 2
    assert next(out1) == 4
    assert count == 3
    assert next(out1) == 5
    assert count == 4


def test_broadcast_trick():
    a = nxp.ones((10, 10), dtype=nxp.int8)
    b = broadcast_trick(nxp.ones)((10, 10), dtype=nxp.int8)

    assert_array_equal(a, b)
    assert a.nbytes == 100
    assert b.base.nbytes == 1

    a = nxp.ones((), dtype=nxp.int8)
    b = broadcast_trick(nxp.ones)((), dtype=nxp.int8)
    assert_array_equal(a, b)


def test_normalize_shape():
    assert normalize_shape(2) == (2,)
    assert normalize_shape((2,)) == (2,)
    assert normalize_shape((2, 0)) == (2, 0)
    assert normalize_shape((2, 3)) == (2, 3)

    with pytest.raises(TypeError):
        normalize_shape(None)
