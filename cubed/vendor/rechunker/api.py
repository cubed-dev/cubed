"""User-facing functions."""
from collections import defaultdict

import zarr

import cubed.vendor.dask

from .algorithm import rechunking_plan
from .types import ArrayProxy, CopySpec


def _shape_dict_to_tuple(dims, shape_dict):
    # convert a dict of shape
    shape = [shape_dict[dim] for dim in dims]
    return tuple(shape)


def _get_dims_from_zarr_array(z_array):
    # use Xarray convention
    # http://xarray.pydata.org/en/stable/internals.html#zarr-encoding-specification
    return z_array.attrs["_ARRAY_DIMENSIONS"]


def _encode_zarr_attributes(attrs):
    from xarray.backends.zarr import encode_zarr_attr_value

    return {k: encode_zarr_attr_value(v) for k, v in attrs.items()}


def _zarr_empty(shape, store_or_group, chunks, dtype, name=None, **kwargs):
    # wrapper that maybe creates the array within a group
    if name is not None:
        assert isinstance(store_or_group, zarr.hierarchy.Group)
        return store_or_group.empty(
            name, shape=shape, chunks=chunks, dtype=dtype, **kwargs
        )
    else:
        return zarr.empty(
            shape, chunks=chunks, dtype=dtype, store=store_or_group, **kwargs
        )


ZARR_OPTIONS = [
    "compressor",
    "filters",
    "order",
    "cache_metadata",
    "cache_attrs",
    "overwrite",
    "write_empty_chunks",
]


def _validate_options(options):
    if not options:
        return
    for o in options:
        if o not in ZARR_OPTIONS:
            raise ValueError(
                f"Zarr options must not include {o} (got {o}={options[o]}). "
                f"Only the following options are supported: {ZARR_OPTIONS}."
            )


def get_dim_chunk(da, dim, target_chunks):
    if dim in target_chunks.keys():
        if target_chunks[dim] > len(da[dim]) or target_chunks[dim] < 0:
            dim_chunk = len(da[dim])
        else:
            dim_chunk = target_chunks[dim]
    else:
        # if not isinstance(da.data, dask.array.Array):
        dim_chunk = len(da[dim])
        # else:
        #     existing_chunksizes = {k: v for k, v in zip(da.dims, da.data.chunksize)}
        #     dim_chunk = existing_chunksizes[dim]
    return dim_chunk


def parse_target_chunks_from_dim_chunks(ds, target_chunks):
    """
    Calculate ``target_chunks`` suitable for ``rechunker.rechunk()`` using chunks defined for
    dataset dimensions (similar to xarray's ``.rechunk()``) .

    - If a dimension is missing from ``target_chunks`` then use the full length from ``ds``.
    - If a chunk in ``target_chunks`` is larger than the full length of the variable in ``ds``,
      then, again, use the full length from the dataset.
    - If a dimension chunk is specified as -1, again, use the full length from the dataset.

    """
    group_chunks = defaultdict(list)

    for var in ds.variables:
        for dim in ds[var].dims:
            group_chunks[var].append(get_dim_chunk(ds[var], dim, target_chunks))

    # rechunk() expects chunks values to be a tuple. So let's convert them
    group_chunks_tuples = {var: tuple(chunks) for (var, chunks) in group_chunks.items()}
    return group_chunks_tuples


def _copy_group_attributes(source, target):
    """Visit every source group and create it on the target and move any attributes found."""

    def _update_group_attrs(name):
        if isinstance(source.get(name), zarr.Group):
            group = target.create_group(name)
            group.attrs.update(source.get(name).attrs)

    source.visit(_update_group_attrs)


def _setup_rechunk(
    source,
    target_chunks,
    max_mem,
    target_store,
    target_options=None,
    temp_store=None,
    temp_options=None,
):
    if temp_options is None:
        temp_options = target_options
    target_options = target_options or {}
    temp_options = temp_options or {}

    # import xarray dynamically since it is not a required dependency
    try:
        import xarray
        from xarray.backends.zarr import (
            DIMENSION_KEY,
            encode_zarr_attr_value,
            encode_zarr_variable,
            extract_zarr_variable_encoding,
        )
        from xarray.conventions import encode_dataset_coordinates
    except ImportError:
        xarray = None

    if xarray and isinstance(source, xarray.Dataset):
        import dask.array

        if not isinstance(target_chunks, dict):
            raise ValueError(
                "You must specify ``target-chunks`` as a dict when rechunking a dataset."
            )

        variables, attrs = encode_dataset_coordinates(source)
        attrs = _encode_zarr_attributes(attrs)

        if temp_store is not None:
            temp_group = zarr.group(temp_store)
        else:
            temp_group = None
        target_group = zarr.group(target_store)
        target_group.attrs.update(attrs)

        # if ``target_chunks`` is specified per dimension (xarray ``.rechunk`` style),
        # parse chunks for each coordinate/variable
        if all([k in source.dims for k in target_chunks.keys()]):
            # ! We can only apply this when all keys are indeed dimension, otherwise it falls back to the old method
            target_chunks = parse_target_chunks_from_dim_chunks(source, target_chunks)

        copy_specs = []
        for name, variable in variables.items():
            # This isn't strictly necessary because a shallow copy
            # also occurs in `encode_dataset_coordinates` but do it
            # anyways in case the coord encoding function changes
            variable = variable.copy()

            # Update the array encoding with provided options and apply it;
            # note that at this point the `options` may contain any valid property
            # applicable for the `encoding` parameter in Dataset.to_zarr other than "chunks"
            options = target_options.get(name, {})
            if "chunks" in options:
                raise ValueError(
                    f"Chunks must be provided in ``target_chunks`` rather than options (variable={name})"
                )
            variable.encoding.update(options)
            variable = encode_zarr_variable(variable)

            # Extract the array encoding to get a default chunking, a step
            # which will also ensure that the target chunking is compatible
            # with the current chunking (only necessary for on-disk arrays)
            variable_encoding = extract_zarr_variable_encoding(
                variable, raise_on_invalid=False, name=name
            )
            variable_chunks = target_chunks.get(name, variable_encoding["chunks"])
            if isinstance(variable_chunks, dict):
                variable_chunks = _shape_dict_to_tuple(variable.dims, variable_chunks)

            # Restrict options to only those that are specific to zarr and
            # not managed internally
            options = {k: v for k, v in options.items() if k in ZARR_OPTIONS}
            _validate_options(options)

            # Extract array attributes along with reserved property for
            # xarray dimension names
            variable_attrs = _encode_zarr_attributes(variable.attrs)
            variable_attrs[DIMENSION_KEY] = encode_zarr_attr_value(variable.dims)

            copy_spec = _setup_array_rechunk(
                dask.array.asarray(variable),
                variable_chunks,
                max_mem,
                target_group,
                target_options=options,
                temp_store_or_group=temp_group,
                temp_options=options,
                name=name,
            )
            copy_spec.write.array.attrs.update(variable_attrs)  # type: ignore
            copy_specs.append(copy_spec)

        return copy_specs, temp_group, target_group

    elif isinstance(source, zarr.hierarchy.Group):
        if not isinstance(target_chunks, dict):
            raise ValueError(
                "You must specify ``target-chunks`` as a dict when rechunking a group."
            )

        if temp_store is not None:
            temp_group = zarr.group(temp_store)
        else:
            temp_group = None
        target_group = zarr.group(target_store)
        _copy_group_attributes(source, target_group)
        target_group.attrs.update(source.attrs)

        copy_specs = []
        for array_name, array_target_chunks in target_chunks.items():
            copy_spec = _setup_array_rechunk(
                source[array_name],
                array_target_chunks,
                max_mem,
                target_group,
                target_options=target_options.get(array_name),
                temp_store_or_group=temp_group,
                temp_options=temp_options.get(array_name),
                name=array_name,
            )
            copy_specs.append(copy_spec)

        return copy_specs, temp_group, target_group

    # elif isinstance(source, (zarr.core.Array, dask.array.Array)):
    elif isinstance(source, zarr.core.Array):

        copy_spec = _setup_array_rechunk(
            source,
            target_chunks,
            max_mem,
            target_store,
            target_options=target_options,
            temp_store_or_group=temp_store,
            temp_options=temp_options,
        )
        intermediate = copy_spec.intermediate.array
        target = copy_spec.write.array
        return [copy_spec], intermediate, target

    else:
        raise ValueError(
            f"Source must be a Zarr Array, Zarr Group, Dask Array or Xarray Dataset (not {type(source)})."
        )


def _setup_array_rechunk(
    source_array,
    target_chunks,
    max_mem,
    target_store_or_group,
    target_options=None,
    temp_store_or_group=None,
    temp_options=None,
    name=None,
) -> CopySpec:
    _validate_options(target_options)
    _validate_options(temp_options)
    shape = source_array.shape
    # source_chunks = (
    #     source_array.chunksize
    #     if isinstance(source_array, dask.array.Array)
    #     else source_array.chunks
    # )
    source_chunks = source_array.chunks
    dtype = source_array.dtype
    itemsize = dtype.itemsize

    if target_chunks is None:
        # this is just a pass-through copy
        target_chunks = source_chunks

    if isinstance(target_chunks, dict):
        array_dims = _get_dims_from_zarr_array(source_array)
        try:
            target_chunks = _shape_dict_to_tuple(array_dims, target_chunks)
        except KeyError:
            raise KeyError(
                "You must explicitly specify each dimension size in target_chunks. "
                f"Got array_dims {array_dims}, target_chunks {target_chunks}."
            )

    # TODO: rewrite to avoid the hard dependency on dask
    max_mem = cubed.vendor.dask.utils.parse_bytes(max_mem)

    # don't consolidate reads for Dask arrays
    consolidate_reads = isinstance(source_array, zarr.core.Array)
    read_chunks, int_chunks, write_chunks = rechunking_plan(
        shape,
        source_chunks,
        target_chunks,
        itemsize,
        max_mem,
        consolidate_reads=consolidate_reads,
    )

    # create target
    shape = tuple(int(x) for x in shape)  # ensure python ints for serialization
    target_chunks = tuple(int(x) for x in target_chunks)
    int_chunks = tuple(int(x) for x in int_chunks)
    write_chunks = tuple(int(x) for x in write_chunks)

    target_array = _zarr_empty(
        shape,
        target_store_or_group,
        target_chunks,
        dtype,
        name=name,
        **(target_options or {}),
    )
    try:
        target_array.attrs.update(source_array.attrs)
    except AttributeError:
        pass

    if read_chunks == write_chunks:
        int_array = None
    else:
        # do intermediate store
        if temp_store_or_group is None:
            raise ValueError(
                "A temporary store location must be provided{}.".format(
                    f" (array={name})" if name else ""
                )
            )
        int_array = _zarr_empty(
            shape,
            temp_store_or_group,
            int_chunks,
            dtype,
            name=name,
            **(temp_options or {}),
        )

    read_proxy = ArrayProxy(source_array, read_chunks)
    int_proxy = ArrayProxy(int_array, int_chunks)
    write_proxy = ArrayProxy(target_array, write_chunks)
    return CopySpec(read_proxy, int_proxy, write_proxy)
