import pytest

import cubed
import cubed.array_api as xp
from cubed._testing import assert_array_equal
from cubed.storage.store import get_storage_name


# This is less strict than the spec, but is supported by implementations like NumPy
def test_prod_sum_bool():
    a = xp.ones((2,), dtype=xp.bool)
    assert_array_equal(xp.prod(a).compute(), xp.asarray([1], dtype=xp.int64))
    assert_array_equal(xp.sum(a).compute(), xp.asarray([2], dtype=xp.int64))


@pytest.mark.skipif(
    get_storage_name() != "zarr-python",
    reason="object dtype only works on zarr-python",
)
def test_object_dtype():
    a = xp.asarray(["a", "b"], dtype=object, chunks=2)
    cubed.to_zarr(a, store=None)
