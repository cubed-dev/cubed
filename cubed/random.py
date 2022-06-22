import random as pyrandom

import numpy as np
import numpy.array_api as nxp
from dask.array.core import normalize_chunks
from numpy.random import Generator, Philox
from zarr.util import normalize_shape

from cubed.utils import to_chunksize

from .array_api import empty
from .core import map_blocks


def random(size, *, chunks=None, spec=None):
    # Create an initial empty array with the same chunk structure as the
    # desired output, but where each chunk has a single element.
    # Then call map_blocks to generate random numbers for each chunk,
    # and change the chunk size to the desired size.

    shape = normalize_shape(size)
    dtype = nxp.float64
    chunks = normalize_chunks(chunks, shape=shape, dtype=dtype)
    chunksize = to_chunksize(chunks)
    numblocks = tuple(map(len, chunks))
    root_seed = pyrandom.getrandbits(128)

    empty_chunks = (1,) * len(numblocks)
    out = empty(numblocks, chunks=empty_chunks, spec=spec)
    out = map_blocks(
        _random,
        out,
        dtype=dtype,
        chunks=chunks,
        size=chunksize,
        numblocks=numblocks,
        root_seed=root_seed,
    )
    return out


def _random(a, size=None, numblocks=None, root_seed=None, block_id=None):
    stream_id = block_id_to_offset(block_id, numblocks)
    rg = Generator(Philox(key=root_seed + stream_id))
    return rg.random(size)


def block_id_to_offset(block_id, numblocks):
    arr = np.empty(numblocks, dtype=np.int8)
    return sum((np.array(block_id) * arr.strides).tolist())
