import random as pyrandom

import numpy as np
import numpy.array_api as nxp
from numpy.random import Generator, Philox
from zarr.util import normalize_shape

from cubed.core.ops import map_direct
from cubed.vendor.dask.array.core import normalize_chunks


def random(size, *, chunks=None, spec=None):
    """Return random floats in the half-open interval [0.0, 1.0)."""
    shape = normalize_shape(size)
    dtype = nxp.float64
    chunks = normalize_chunks(chunks, shape=shape, dtype=dtype)
    numblocks = tuple(map(len, chunks))
    root_seed = pyrandom.getrandbits(128)

    # no extra memory is projected to be needed since input is an empty array whose
    # memory is never allocated, see https://pythonspeed.com/articles/measuring-memory-python/#phantom-memory
    extra_projected_mem = 0

    return map_direct(
        _random,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        extra_projected_mem=extra_projected_mem,
        spec=spec,
        numblocks=numblocks,
        root_seed=root_seed,
    )


def _random(x, *arrays, numblocks=None, root_seed=None, block_id=None):
    stream_id = block_id_to_offset(block_id, numblocks)
    rg = Generator(Philox(key=root_seed + stream_id))
    return rg.random(x.shape)


def block_id_to_offset(block_id, numblocks):
    arr = np.empty(numblocks, dtype=np.int8)
    return sum((np.array(block_id) * arr.strides).tolist())
