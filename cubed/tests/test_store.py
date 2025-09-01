import numpy as np
import pytest
import zarr
from numpy.testing import assert_array_equal

import cubed
import cubed.array_api as xp

ZARR_PYTHON_V2 = zarr.__version__[0] == "2"


@pytest.mark.skipif(
    ZARR_PYTHON_V2,
    reason="setting an arbitrary Zarr store is not supported for Zarr Python v2",
)
def test_arbitrary_zarr_store():
    store = zarr.storage.MemoryStore()
    spec = cubed.Spec(intermediate_store=store, allowed_mem="100kB")
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
    c = xp.add(a, b)
    assert_array_equal(c, np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]]))

    # check store was used
    z = zarr.open_group(store)
    array_keys = list(z.array_keys())
    assert len(array_keys) == 1
    assert array_keys[0].startswith("array-")
