import numpy as np
from tlz import concat, merge, unique

from cubed.vendor.dask.array.gufunc import _parse_gufunc_signature


def apply_gufunc(
    func,
    signature,
    *args,
    axes=None,
    axis=None,
    output_dtypes=None,
    output_sizes=None,
    vectorize=None,
    **kwargs,
):
    """
    Apply a generalized ufunc or similar python function to arrays.

    This is a cutdown version of the
    `equivalent function <https://docs.dask.org/en/stable/generated/dask.array.gufunc.apply_gufunc.html>`_
    in Dask. Refer there for usage information.

    Current limitations: ``keepdims``, and ``allow_rechunk`` are not supported;
    and multiple outputs are not supported.

    Cubed assumes that ``func`` will allocate a new output array. However, if it allocates more memory
    than than, then you need to tell Cubed about it by setting the ``extra_projected_mem`` parameter
    to the amount needed in bytes (per task).
    """

    # Currently the following parameters cannot be changed
    # keepdims = False
    allow_rechunk = False

    # based on dask's apply_gufunc

    # Input processing:

    # Signature
    if not isinstance(signature, str):
        raise TypeError("`signature` has to be of type string")
    input_coredimss, output_coredimss = _parse_gufunc_signature(signature)

    # Determine nout: nout = None for functions of one direct return; nout = int for return tuples
    nout = None if not isinstance(output_coredimss, list) else len(output_coredimss)

    if nout is not None:
        raise NotImplementedError(
            "Multiple outputs are not yet supported, see https://github.com/cubed-dev/cubed/issues/69"
        )

    # Vectorize function, if required
    if vectorize:
        otypes = output_dtypes
        func = np.vectorize(func, signature=signature, otypes=otypes)

    # Miscellaneous
    if output_sizes is None:
        output_sizes = {}

    # Axes
    # input_axes, output_axes = _validate_normalize_axes(
    #     axes, axis, keepdims, input_coredimss, output_coredimss
    # )

    # Main code:

    # Cast all input arrays to cubed
    # Use a spec if there is one. Note that all args have to have the same spec, and
    # this will be checked later when constructing the plan (see check_array_specs).
    from cubed.array_api.creation_functions import asarray

    specs = [a.spec for a in args if hasattr(a, "spec")]
    spec = specs[0] if len(specs) > 0 else None
    args = [asarray(a, spec=spec) for a in args]

    if len(input_coredimss) != len(args):
        raise ValueError(
            "According to `signature`, `func` requires %d arguments, but %s given"
            % (len(input_coredimss), len(args))
        )

    # Note (cubed): since we don't support allow_rechunk=True, there is no need to transpose args (and outputs back again)

    # Assess input args for loop dims
    input_shapes = [a.shape for a in args]
    input_chunkss = [a.chunks for a in args]
    num_loopdims = [len(s) - len(cd) for s, cd in zip(input_shapes, input_coredimss)]
    max_loopdims = max(num_loopdims) if num_loopdims else None
    core_input_shapes = [
        dict(zip(icd, s[n:]))
        for s, n, icd in zip(input_shapes, num_loopdims, input_coredimss)
    ]
    core_shapes = merge(*core_input_shapes)
    core_shapes.update(output_sizes)

    loop_input_dimss = [
        tuple("__loopdim%d__" % d for d in range(max_loopdims - n, max_loopdims))
        for n in num_loopdims
    ]
    input_dimss = [lp + c for lp, c in zip(loop_input_dimss, input_coredimss)]

    loop_output_dims = max(loop_input_dimss, key=len) if loop_input_dimss else tuple()

    # Assess input args for same size and chunk sizes
    # Collect sizes and chunksizes of all dims in all arrays
    dimsizess = {}
    chunksizess = {}
    for dims, shape, chunksizes in zip(input_dimss, input_shapes, input_chunkss):
        for dim, size, chunksize in zip(dims, shape, chunksizes):
            dimsizes = dimsizess.get(dim, [])
            dimsizes.append(size)
            dimsizess[dim] = dimsizes
            chunksizes_ = chunksizess.get(dim, [])
            chunksizes_.append(chunksize)
            chunksizess[dim] = chunksizes_
    # Assert correct partitioning, for case:
    for dim, sizes in dimsizess.items():
        # Check that the arrays have same length for same dimensions or dimension `1`
        if set(sizes) | {1} != {1, max(sizes)}:
            raise ValueError(f"Dimension `'{dim}'` with different lengths in arrays")
        if not allow_rechunk:
            chunksizes = chunksizess[dim]
            # Check if core dimensions consist of only one chunk
            if (dim in core_shapes) and (chunksizes[0][0] < core_shapes[dim]):
                raise ValueError(
                    "Core dimension `'{}'` consists of multiple chunks. To fix, rechunk into a single \
chunk along this dimension or set `allow_rechunk=True`, but beware that this may increase memory usage \
significantly.".format(
                        dim
                    )
                )
            # Check if loop dimensions consist of same chunksizes, when they have sizes > 1
            relevant_chunksizes = list(
                unique(c for s, c in zip(sizes, chunksizes) if s > 1)
            )
            if len(relevant_chunksizes) > 1:
                raise ValueError(
                    f"Dimension `'{dim}'` with different chunksize present"
                )

    # Apply function - use blockwise here
    arginds = list(concat(zip(args, input_dimss)))

    from cubed.core.ops import blockwise

    # Note (cubed): use blockwise on all output dimensions, not just loop_output_dims like in original
    out_ind = loop_output_dims + output_coredimss

    return blockwise(
        func, out_ind, *arginds, dtype=output_dtypes, new_axes=output_sizes, **kwargs
    )
