import numpy as np
import pytest
from numpy.testing import assert_array_equal

from cubed.primitive.reshape import reshape_chunks
from cubed.tests.utils import create_zarr


@pytest.mark.parametrize(
    "shape, chunks, new_shape, new_chunks",
    [
        ((12,), 4, (3, 4), (1, 4)),
        ((1,), (1,), (), ()),
    ],
)
def test_reshape_chunks(tmp_path, shape, chunks, new_shape, new_chunks):
    x = np.random.randint(10, size=shape)
    source = create_zarr(
        x,
        dtype=int,
        chunks=chunks,
        store=tmp_path / "source.zarr",
    )
    target = reshape_chunks(source, new_shape, new_chunks)

    assert target.shape == new_shape
    assert target.chunks == new_chunks
    assert_array_equal(target[...], np.reshape(x, new_shape))


def test_reshape_chunks_errors(tmp_path):
    x = np.random.randint(10, size=(12,))
    source = create_zarr(
        x,
        dtype=int,
        chunks=(3,),
        store=tmp_path / "source.zarr",
    )

    with pytest.raises(ValueError):
        # different size
        reshape_chunks(source, (100,), (10,))

    with pytest.raises(ValueError):
        # chunks don't divide exactly
        reshape_chunks(source, (3, 4), (3, 3))
