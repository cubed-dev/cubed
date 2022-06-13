import numpy as np
import pytest
from numpy.testing import assert_array_equal

from cubed.primitive.views.squeeze import squeeze
from cubed.tests.utils import create_zarr


@pytest.mark.parametrize("axis", [0, -1, (0, -1)])
def test_squeeze(tmp_path, axis):
    x = np.arange(10)[None, :, None, None]
    chunks = (1, 3, 1, 1)
    source = create_zarr(
        x,
        dtype=int,
        chunks=chunks,
        store=tmp_path / "source.zarr",
    )
    target = squeeze(source, axis=axis)

    assert_array_equal(target[...], np.squeeze(x, axis=axis))


def test_squeeze_errors(tmp_path):
    x = np.arange(10)[None, :, None, None]
    chunks = (1, 3, 1, 1)
    source = create_zarr(
        x,
        dtype=int,
        chunks=chunks,
        store=tmp_path / "source.zarr",
    )

    with pytest.raises(ValueError):
        # axis must be a singleton
        squeeze(source, axis=1)

    with pytest.raises(TypeError):
        # axis can't be None
        squeeze(source, axis=None)
