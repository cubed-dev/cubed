"""
Tests for deterministic rechunk data generation and validation.

The _det_block / make_deterministic_source helpers are copied from
examples/rechunk-bench.py; keeping them here makes the test self-contained
and avoids a library dependency on example-script internals.
"""

import math

import numpy as np

import cubed
from cubed._testing import assert_array_equal


def _det_block(block, block_id, _shape, _chunks):
    strides = [math.prod(_shape[i + 1 :]) for i in range(len(_shape))]
    flat = np.zeros(block.shape, dtype=np.int64)
    for ax, (stride, cs) in enumerate(zip(strides, _chunks)):
        origin = block_id[ax] * cs
        idx = np.arange(origin, origin + block.shape[ax], dtype=np.int64)
        flat += (
            idx.reshape(tuple(-1 if j == ax else 1 for j in range(len(_shape))))
            * stride
        )
    return (flat % (2**31)).astype(np.int32)


def make_deterministic_source(shape, chunks):
    """Return a lazy int32 array where each element equals its flat index modulo 2**31."""
    template = cubed.zeros(shape, dtype=np.int32, chunks=chunks)
    return cubed.map_blocks(
        _det_block, template, dtype=np.int32, _shape=shape, _chunks=chunks
    )


def test_deterministic_source_values():
    shape = (10, 8, 6)
    chunks = (5, 4, 3)
    result = make_deterministic_source(shape, chunks).compute()

    assert result.shape == shape
    assert result.dtype == np.int32

    strides = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
    for idx in [(0, 0, 0), (0, 0, 1), (1, 0, 0), (5, 4, 3), (9, 7, 5)]:
        flat = sum(i * s for i, s in zip(idx, strides))
        assert result[idx] == flat % (2**31)


def test_deterministic_rechunk():
    shape = (10, 8, 6)
    source_chunks = (5, 4, 3)
    target_chunks = (10, 2, 2)

    rechunked = make_deterministic_source(shape, source_chunks).rechunk(target_chunks)
    expected = make_deterministic_source(shape, target_chunks)
    assert_array_equal(rechunked.compute(), expected.compute())
