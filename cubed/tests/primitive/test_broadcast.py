import numpy as np
import pytest
from numpy.testing import assert_array_equal

from cubed.primitive import broadcast_to
from cubed.tests.utils import create_zarr


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


@pytest.mark.parametrize(
    "shape, chunks, new_shape, new_chunks, new_chunks_expected",
    [
        ((5, 1, 6), (3, 1, 3), (5, 3, 6), (3, 1, 3), (3, 1, 3)),
        ((5, 1, 6), (3, 1, 3), (5, 3, 6), (3, 3, 3), (3, 3, 3)),
        ((5, 1, 6), (3, 1, 3), (2, 5, 3, 6), (1, 3, 1, 3), (1, 3, 1, 3)),
    ],
)
def test_broadcast_to_chunks(
    tmp_path, shape, chunks, new_shape, new_chunks, new_chunks_expected
):
    x = np.random.randint(10, size=shape)
    source = create_zarr(
        x,
        dtype=int,
        chunks=chunks,
        store=tmp_path / "source.zarr",
    )
    target = broadcast_to(source, new_shape, chunks=new_chunks)

    assert target.shape == new_shape
    assert target.chunks == new_chunks_expected
    assert_array_equal(target[:], np.broadcast_to(x, new_shape))


def test_broadcast_to_errors(tmp_path):
    x = np.random.randint(10, size=(5, 1, 6))
    source = create_zarr(
        x,
        dtype=int,
        chunks=(3, 1, 3),
        store=tmp_path / "source.zarr",
    )

    with pytest.raises(ValueError):
        broadcast_to(source, (2, 1, 6))
    with pytest.raises(ValueError):
        broadcast_to(source, (3,))

    # can't rechunk existing (non-broadcast) dimensions
    with pytest.raises(ValueError):
        broadcast_to(source, source.shape, chunks=(2, 1, 3))
