import numpy as np
import pytest
from numpy.testing import assert_array_equal

from barry.primitive import broadcast_to
from barry.tests.utils import create_zarr


@pytest.mark.parametrize(
    "shape, chunks, new_shape, new_chunks_expected",
    [
        ((5, 1, 6), (3, 1, 3), (5, 0, 6), (3, 1, 3)),
        ((5, 1, 6), (3, 1, 3), (5, 4, 6), (3, 1, 3)),
        ((5, 1, 6), (3, 1, 3), (2, 5, 1, 6), (1, 3, 1, 3)),
        ((5, 1, 6), (3, 1, 3), (3, 4, 5, 4, 6), (1, 1, 3, 1, 3)),
    ],
)
def test_broadcast_to(tmp_path, shape, chunks, new_shape, new_chunks_expected):
    x = np.random.randint(10, size=shape)
    source = create_zarr(
        x,
        dtype=int,
        chunks=chunks,
        store=tmp_path / "source.zarr",
    )
    target = broadcast_to(source, new_shape)

    assert target.shape == new_shape
    assert target.chunks == new_chunks_expected
    assert_array_equal(target[:], np.broadcast_to(x, new_shape))
