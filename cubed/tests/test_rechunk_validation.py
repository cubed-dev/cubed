"""
Tests for deterministic rechunk data generation and validation.

The helpers here are copied from examples/rechunk-bench.py; keeping them
self-contained avoids a library dependency on example-script internals.

Two data generation strategies are tested:

- Sequential: element value = flat_index % 2**31.  Simple but highly
  compressible (constant delta between adjacent elements).

- Hashed: element value = wang_hash(flat_index), reinterpreted as int32.
  Bijective hash breaks the sequential pattern; output is effectively
  random bits and therefore incompressible, matching float32 random data.
"""

import math

import numpy as np

import cubed
import cubed.array_api as xp
from cubed.utils import normalize_chunks, to_chunksize

# ── Sequential (flat-index) ──────────────────────────────────────────────────


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
    normalized = normalize_chunks(chunks, shape=shape, dtype=np.int32)
    return cubed.map_blocks(
        _det_block,
        dtype=np.int32,
        chunks=normalized,
        _shape=shape,
        _chunks=to_chunksize(normalized),
    )


# ── Hashed (Wang hash) ───────────────────────────────────────────────────────


def _wang_hash(n):
    """Bijective 32-bit Wang hash applied elementwise to a numpy array."""
    n = n.astype(np.uint32)
    n = (n ^ np.uint32(61)) ^ (n >> np.uint32(16))
    n = n + (n << np.uint32(3))
    n = n ^ (n >> np.uint32(4))
    n = n * np.uint32(0x27D4EB2D)
    n = n ^ (n >> np.uint32(15))
    return n.view(np.int32)


def _det_block_hashed(block, block_id, _shape, _chunks):
    strides = [math.prod(_shape[i + 1 :]) for i in range(len(_shape))]
    flat = np.zeros(block.shape, dtype=np.int64)
    for ax, (stride, cs) in enumerate(zip(strides, _chunks)):
        origin = block_id[ax] * cs
        idx = np.arange(origin, origin + block.shape[ax], dtype=np.int64)
        flat += (
            idx.reshape(tuple(-1 if j == ax else 1 for j in range(len(_shape))))
            * stride
        )
    return _wang_hash(flat.astype(np.uint32))


def make_hashed_source(shape, chunks):
    """Return a lazy int32 array where each element is wang_hash(flat_index)."""
    normalized = normalize_chunks(chunks, shape=shape, dtype=np.int32)
    return cubed.map_blocks(
        _det_block_hashed,
        dtype=np.int32,
        chunks=normalized,
        _shape=shape,
        _chunks=to_chunksize(normalized),
    )


# ── Tests: sequential ────────────────────────────────────────────────────────


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


def test_deterministic_rechunk(tmp_path):
    shape = (10, 8, 6)
    source_chunks = (5, 4, 3)
    target_chunks = (10, 2, 2)

    rechunked = make_deterministic_source(shape, source_chunks).rechunk(target_chunks)
    cubed.to_zarr(rechunked, store=tmp_path / "target")

    target = cubed.from_zarr(tmp_path / "target")
    expected = make_deterministic_source(shape, target_chunks)
    assert bool(xp.all(xp.equal(target, expected)).compute())


# ── Tests: hashed ────────────────────────────────────────────────────────────


def test_hashed_source_values():
    shape = (10, 8, 6)
    chunks = (5, 4, 3)
    result = make_hashed_source(shape, chunks).compute()

    assert result.shape == shape
    assert result.dtype == np.int32

    # check sampled elements match wang_hash(flat_index)
    strides = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
    for idx in [(0, 0, 0), (0, 0, 1), (1, 0, 0), (5, 4, 3), (9, 7, 5)]:
        flat = sum(i * s for i, s in zip(idx, strides))
        expected = _wang_hash(np.array([flat], dtype=np.uint32))[0]
        assert result[idx] == expected

    # adjacent elements should differ substantially (no constant-delta pattern)
    flat_vals = result.ravel().astype(np.int64)
    deltas = np.diff(flat_vals)
    assert not np.all(deltas == deltas[0]), "hashed values must not have constant delta"


def test_hashed_rechunk(tmp_path):
    shape = (10, 8, 6)
    source_chunks = (5, 4, 3)
    target_chunks = (10, 2, 2)

    rechunked = make_hashed_source(shape, source_chunks).rechunk(target_chunks)
    cubed.to_zarr(rechunked, store=tmp_path / "target")

    target = cubed.from_zarr(tmp_path / "target")
    expected = make_hashed_source(shape, target_chunks)
    assert bool(xp.all(xp.equal(target, expected)).compute())
