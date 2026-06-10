import math
import random as pyrandom

import numpy as np

from cubed.backend_array_api import namespace as nxp
from cubed.backend_array_api import numpy_array_to_backend_array
from cubed.core.ops import map_blocks
from cubed.utils import (
    block_id_to_offset,
    normalize_chunks,
    normalize_shape,
    to_chunksize,
)


def random(size, *, dtype=nxp.float64, chunks=None, spec=None):
    """Return random floats in the half-open interval [0.0, 1.0)."""
    shape = normalize_shape(size)
    chunks = normalize_chunks(chunks, shape=shape, dtype=dtype)
    numblocks = tuple(map(len, chunks))
    root_seed = pyrandom.getrandbits(128)
    extra_func_kwargs = dict(dtype=dtype)
    return map_blocks(
        _random,
        dtype=dtype,
        chunks=chunks,
        spec=spec,
        numblocks=numblocks,
        root_seed=root_seed,
        extra_func_kwargs=extra_func_kwargs,
    )


def _random(x, numblocks=None, root_seed=None, dtype=nxp.float64, block_id=None):
    # import as needed to avoid slow 'import cubed'
    from numpy.random import Generator, Philox

    stream_id = block_id_to_offset(block_id, numblocks)
    rg = Generator(Philox(key=root_seed + stream_id))
    out = rg.random(x.shape, dtype=dtype)
    out = numpy_array_to_backend_array(out)
    return out


def integers(shape, *, dtype=nxp.int32, chunks=None, spec=None):
    """Return a deterministic integer array where element[i] = wang_hash(flat_index(i)).

    Each element's value depends only on its position in the array, not on chunking.
    This means the same logical array can be regenerated at any chunk layout and
    compared element-wise — making it suitable for rechunk validation and I/O benchmarks.

    The Wang hash produces output with no constant-delta pattern between adjacent
    elements, so the array is effectively incompressible under shuffle-based codecs
    (comparable to float32 random data).

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the array.
    dtype : dtype, optional
        Integer dtype for the output. Default is int32.
    chunks : tuple of ints, optional
        Chunk shape. Defaults to the array shape (single chunk).
    spec : cubed.Spec, optional
        The spec to use for the computation.

    Returns
    -------
    cubed.Array
        A lazy integer array backed by wang_hash(flat_index).
    """
    shape = normalize_shape(shape)
    chunks = normalize_chunks(chunks, shape=shape, dtype=dtype)
    chunksize = to_chunksize(chunks)
    return map_blocks(
        _integers,
        dtype=dtype,
        chunks=chunks,
        spec=spec,
        _shape=shape,
        _chunksize=chunksize,
    )


def _integers(x, _shape, _chunksize, block_id=None):
    strides = [math.prod(_shape[i + 1 :]) for i in range(len(_shape))]
    flat = nxp.zeros(x.shape, dtype=nxp.int64)
    for ax, (stride, cs) in enumerate(zip(strides, _chunksize)):
        origin = block_id[ax] * cs
        idx = nxp.arange(origin, origin + x.shape[ax], dtype=nxp.int64)
        flat = (
            flat
            + nxp.reshape(idx, tuple(-1 if j == ax else 1 for j in range(len(_shape))))
            * stride
        )
    # _wang_hash uses numpy-specific .view(); convert and convert back
    hashed = _wang_hash(np.asarray(flat, dtype=np.uint32))
    return numpy_array_to_backend_array(hashed).astype(x.dtype)


def _wang_hash(n):
    """Bijective 32-bit Wang hash applied elementwise to a uint32 numpy array."""
    n = (n ^ np.uint32(61)) ^ (n >> np.uint32(16))
    n = n + (n << np.uint32(3))
    n = n ^ (n >> np.uint32(4))
    n = n * np.uint32(0x27D4EB2D)
    n = n ^ (n >> np.uint32(15))
    return n.view(np.int32)
