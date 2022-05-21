import numpy as np
import numpy.array_api as nxp
from dask.array.core import normalize_chunks
from zarr.util import normalize_shape

from .array_api import ones
from .core import map_blocks


def _random(a, size=None):
    # ignore 'a' - it is a just a placeholder (until used as a seed)
    return np.random.random(size)


def random(size, *, chunks=None, spec=None):
    # Create an initial array of ones with the same chunk structure as the
    # desired output, but where each chunk has a single element.
    # Then call map_blocks to generate random numbers for each chunk,
    # and change the chunk size to the desired size.
    # TODO: support seed (the initial array has one seed per chunk)
    shape = normalize_shape(size)
    dtype = nxp.float64
    chunks = normalize_chunks(chunks, shape=shape, dtype=dtype)
    chunksize = tuple(max(c) for c in chunks)
    numblocks = tuple(map(len, chunks))
    ones_chunks = (1,) * len(numblocks)
    out = ones(numblocks, chunks=ones_chunks, spec=spec)
    out = map_blocks(_random, out, dtype=dtype, chunks=chunks, size=chunksize)
    return out
