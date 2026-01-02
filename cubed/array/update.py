import ndindex

from cubed.array_api.array_object import Array
from cubed.array_api.manipulation_functions import concat
from cubed.core.ops import _create_zarr_indexer, _store_array, map_blocks
from cubed.storage.store import is_storage_array


def append(a, array, /, *, axis=0):
    """Append array to Zarr array a along axis"""

    z = a._zarray

    if not is_storage_array(z):
        raise ValueError("Array must be a Zarr array to perform in-place set operation")

    # these are checks from concat
    arrays = [z, array]

    if len({a.dtype for a in arrays}) > 1:
        raise ValueError("append inputs must all have the same dtype")

    ndim = z.ndim
    if not all(
        i == axis or all(x.shape[i] == arrays[0].shape[i] for x in arrays)
        for i in range(ndim)
    ):
        raise ValueError(
            f"all the input array dimensions except for the append axis must match exactly: {[x.shape for x in arrays]}"
        )

    shape_axis = z.shape[axis]
    offset = shape_axis % z.chunks[axis]

    # find region to write update array to
    region = tuple(
        slice(z.shape[i] - offset, z.shape[i] + s) if i == axis else slice(0, s)
        for i, s in enumerate(array.shape)
    )

    # find new shape of zarr array (z)
    new_shape = tuple(
        s + array.shape[axis] if i == axis else s for i, s in enumerate(z.shape)
    )
    z.resize(new_shape)

    if offset == 0:
        # note the returned array should only really have compute called on it
        # TODO: change to_zarr to have a compute arg, which if False returns the array
        return _store_array(array, z, region=region)
    else:
        # TODO: fix https://github.com/cubed-dev/cubed/issues/414 to avoid a write to disk
        idx = (slice(None),) * axis + (
            slice(shape_axis - offset, shape_axis),
            Ellipsis,
        )
        end_part = a[idx]
        array_with_end_part = concat([end_part, array], axis=axis)

        return _store_array(array_with_end_part, z, region=region)


def set_scalar(source: "Array", key, value):
    """Set scalar value on Zarr array indexing by key."""

    # check that value is a scalar, so we don't have to worry about chunk selection, broadcasting, etc
    if isinstance(value, Array):
        raise NotImplementedError("Only scalar values are supported for set")

    if not is_storage_array(source._zarray):
        raise ValueError("Array must be a Zarr array to perform in-place set operation")

    target = source._zarray  # Note: in place!
    chunks = target.chunks
    idx = ndindex.ndindex(key)
    idx = idx.expand(source.shape)
    selection = idx.raw
    indexer = _create_zarr_indexer(selection, source.shape, source.chunksize)
    output_blocks = map(
        lambda chunk_projection: list(chunk_projection[0]), list(indexer)
    )
    chunk_selections = {cp.chunk_coords: cp.chunk_selection for cp in indexer}

    # note the returned array should only really have compute called on it
    return map_blocks(
        _set,
        source,
        dtype=source.dtype,
        chunks=chunks,
        target_store=target,
        output_blocks=output_blocks,
        value=value,
        chunk_selections=chunk_selections,
    )


def _set(a, value=None, chunk_selections=None, block_id=None):
    a[chunk_selections[block_id]] = value
    return a


def set2_(source: "Array", key, value):
    """Set value on Zarr array indexing by key."""

    # assume that value is a scalar, so we don't have to worry about chunk selection, broadcasting, etc
    if isinstance(value, Array):
        raise NotImplementedError("Only scalar values are supported for set")

    if not is_storage_array(source._zarray):
        raise ValueError("Array must be a Zarr array to perform in-place set operation")

    idx = ndindex.ndindex(key)
    idx = idx.expand(source.shape)
    selection = idx.raw
    print(selection, source.shape, source.chunksize)

    # idea: find region (chunks) that need updating
    chunk_size = ndindex.ChunkSize(source.chunksize)
    region = chunk_size.containing_block(selection, source.shape)
    print(region)

    subchunks = chunk_size.as_subchunks(selection, source.shape)
    for c in subchunks:
        # c is the slicing to get a chunk, and idx.as_subindex(c) is the slicing of the chunk
        # not quite what we want?
        print(c, idx.as_subindex(c))

    idx_region = ndindex.ndindex(region).expand(source.shape)
    idx_within_region = idx.as_subindex(idx_region)
    print("idx_within_region", idx_within_region)

    # then get source array for the region
    source_region = source[region.raw]

    # update the source region array
    # TODO: use code like set_ but which 1) updates whole array, 2) isn't in-place (Cubed will optimize)
    key0 = idx_within_region.raw
    source_region_updated = set0_(source_region, key0, value)

    print("source_region_updated", source_region_updated)
    print("source", source)
    print("region.raw", region.raw)

    # store the updated source region array back to the original source
    return _store_array(source_region_updated, source, region=region.raw)


def set0_(source: "Array", key, value):
    """Set value on Zarr array indexing by key."""

    idx = ndindex.ndindex(key)
    idx = idx.expand(source.shape)
    selection = idx.raw
    indexer = _create_zarr_indexer(selection, source.shape, source.chunksize)
    chunk_selections = {cp.chunk_coords: cp.chunk_selection for cp in indexer}

    # note the returned array should only really have compute called on it
    return map_blocks(
        _set,
        source,
        dtype=source.dtype,
        chunks=source.chunks,
        value=value,
        chunk_selections=chunk_selections,
    )
