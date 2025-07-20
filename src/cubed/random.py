import random as pyrandom

from numpy.random import Generator, Philox

from cubed.backend_array_api import namespace as nxp
from cubed.backend_array_api import numpy_array_to_backend_array
from cubed.core.ops import map_blocks
from cubed.utils import block_id_to_offset, normalize_shape
from cubed.vendor.dask.array.core import normalize_chunks


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
    stream_id = block_id_to_offset(block_id, numblocks)
    rg = Generator(Philox(key=root_seed + stream_id))
    out = rg.random(x.shape, dtype=dtype)
    out = numpy_array_to_backend_array(out)
    return out
