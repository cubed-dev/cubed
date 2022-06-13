import numpy as np
import pytest
from numpy import AxisError
from numpy.testing import assert_array_equal

from cubed.primitive.views.expand_dims import expand_dims
from cubed.tests.utils import create_zarr


@pytest.mark.parametrize("axis", [0, 1, -1, (0, 1)])
def test_expand_dims(tmp_path, axis):
    x = np.arange(10)
    chunks = (3,)
    source = create_zarr(
        x,
        dtype=int,
        chunks=chunks,
        store=tmp_path / "source.zarr",
    )
    target = expand_dims(source, axis=axis)

    assert_array_equal(target[...], np.expand_dims(x, axis=axis))


def test_expand_dims_errors(tmp_path):
    x = np.arange(10)
    chunks = (3,)
    source = create_zarr(
        x,
        dtype=int,
        chunks=chunks,
        store=tmp_path / "source.zarr",
    )

    with pytest.raises(AxisError):
        # axis must be a singleton
        expand_dims(source, axis=2)

    with pytest.raises(TypeError):
        # axis can't be None
        expand_dims(source, axis=None)
