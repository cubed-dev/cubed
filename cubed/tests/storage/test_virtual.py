from itertools import product
from math import prod

import numpy as np
import pytest

from cubed.storage.virtual import virtual_empty, virtual_offsets


@pytest.mark.parametrize(
    "shape,chunks,index",
    [
        ((3,), (2,), 2),
        ((3, 2), (2, 1), (2, 1)),
        ((3, 2), (2, 1), (2, slice(0, 1))),
        ((3, 2), (2, 1), (slice(1, 3), 1)),
        ((3, 2), (2, 1), (slice(1, 3), slice(0, 1))),
    ],
)
def test_virtual_empty(shape, chunks, index):
    # array contents can be any uninitialized values, so
    # just check shapes not values
    v_empty = virtual_empty(shape, dtype=np.int32, chunks=chunks)
    empty = np.empty(shape, dtype=np.int32)
    assert v_empty[index].shape == empty[index].shape
    assert v_empty[...].shape == empty[...].shape


@pytest.mark.parametrize("shape", [(), (3,), (3, 2)])
def test_virtual_offsets(shape):
    v_offsets = virtual_offsets(shape)
    offsets = np.arange(prod(shape)).reshape(shape, order="C")
    for t in product(*(range(n) for n in shape)):
        assert v_offsets[t] == offsets[t]

    # test some length 1 slices
    if len(shape) == 1:
        assert v_offsets[1:2] == offsets[1:2]
    elif len(shape) == 2:
        assert v_offsets[1:2, 0:1] == offsets[1:2, 0:1]


def test_virtual_offsets_fails():
    with pytest.raises(NotImplementedError):
        v_offsets = virtual_offsets((3,))
        v_offsets[0:2]

    with pytest.raises(NotImplementedError):
        v_offsets = virtual_offsets((3, 2))
        v_offsets[0:2, 1]
