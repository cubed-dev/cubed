import re

from tlz import concat, merge, unique

from dask.array.gufunc import _parse_gufunc_signature, _validate_normalize_axes


def apply_gufunc(
    func,
    signature,
    *args,
    axes=None,
    axis=None,
    keepdims=False,
    output_dtypes=None,
    output_sizes=None,
    vectorize=None,
    allow_rechunk=False,
    **kwargs,
):
    """
    Apply a generalized ufunc or similar python function to arrays.

    ``signature`` determines if the function consumes or produces core
    dimensions. The remaining dimensions in given input arrays (``*args``)
    are considered loop dimensions and are required to broadcast
    naturally against each other.

    In other terms, this function is like ``np.vectorize``, but for
    the blocks of cubed arrays. If the function itself shall also
    be vectorized use ``vectorize=True`` for convenience.

    Parameters
    ----------
    func : callable
        Function to call like ``func(*args, **kwargs)`` on input arrays
        (``*args``) that returns an array or tuple of arrays. If multiple
        arguments with non-matching dimensions are supplied, this function is
        expected to vectorize (broadcast) over axes of positional arguments in
        the style of NumPy universal functions [1]_ (if this is not the case,
        set ``vectorize=True``). If this function returns multiple outputs,
        ``output_core_dims`` has to be set as well.
    signature: string
        Specifies what core dimensions are consumed and produced by ``func``.
        According to the specification of numpy.gufunc signature [2]_
    *args : numeric
        Input arrays or scalars to the callable function.
    axes: List of tuples, optional, keyword only
        A list of tuples with indices of axes a generalized ufunc should operate on.
        For instance, for a signature of ``"(i,j),(j,k)->(i,k)"`` appropriate for
        matrix multiplication, the base elements are two-dimensional matrices
        and these are taken to be stored in the two last axes of each argument. The
        corresponding axes keyword would be ``[(-2, -1), (-2, -1), (-2, -1)]``.
        For simplicity, for generalized ufuncs that operate on 1-dimensional arrays
        (vectors), a single integer is accepted instead of a single-element tuple,
        and for generalized ufuncs for which all outputs are scalars, the output
        tuples can be omitted.
    axis: int, optional, keyword only
        A single axis over which a generalized ufunc should operate. This is a short-cut
        for ufuncs that operate over a single, shared core dimension, equivalent to passing
        in axes with entries of (axis,) for each single-core-dimension argument and ``()`` for
        all others. For instance, for a signature ``"(i),(i)->()"``, it is equivalent to passing
        in ``axes=[(axis,), (axis,), ()]``.
    keepdims: bool, optional, keyword only
        If this is set to True, axes which are reduced over will be left in the result as
        a dimension with size one, so that the result will broadcast correctly against the
        inputs. This option can only be used for generalized ufuncs that operate on inputs
        that all have the same number of core dimensions and with outputs that have no core
        dimensions , i.e., with signatures like ``"(i),(i)->()"`` or ``"(m,m)->()"``.
        If used, the location of the dimensions in the output can be controlled with axes
        and axis.
    output_dtypes : Optional, dtype or list of dtypes, keyword only
        Valid numpy dtype specification or list thereof.
        If not given, a call of ``func`` with a small set of data
        is performed in order to try to automatically determine the
        output dtypes.
    output_sizes : dict, optional, keyword only
        Optional mapping from dimension names to sizes for outputs. Only used if
        new core dimensions (not found on inputs) appear on outputs.
    vectorize: bool, keyword only
        If set to ``True``, ``np.vectorize`` is applied to ``func`` for
        convenience. Defaults to ``False``.
    **kwargs : dict
        Extra keyword arguments to pass to `func`

    Returns
    -------
    Single cubed.Array or tuple of cubed.Array
    """
    # based on dask

    # Input processing:
    ## Signature
    if not isinstance(signature, str):
        raise TypeError("`signature` has to be of type string")
    # NumPy versions before https://github.com/numpy/numpy/pull/19627
    # would not ignore whitespace characters in `signature` like they
    # are supposed to. We remove the whitespace here as a workaround.
    signature = re.sub(r"\s+", "", signature)
    input_coredimss, output_coredimss = _parse_gufunc_signature(signature)

    ## Determine nout: nout = None for functions of one direct return; nout = int for return tuples
    nout = None if not isinstance(output_coredimss, list) else len(output_coredimss)

    if nout is not None:
        raise NotImplementedError("Multiple outputs are not yet supported, see GH issue #69")

    # TODO deal with meta business

    ## Vectorize function, if required
    if vectorize:
        raise NotImplementedError()
        otypes = [x.dtype for x in meta] if isinstance(meta, tuple) else [meta.dtype]
        func = np.vectorize(func, signature=signature, otypes=otypes)

    ## Miscellaneous
    if output_sizes is None:
        output_sizes = {}

    ## Axes
    input_axes, output_axes = _validate_normalize_axes(
        axes, axis, keepdims, input_coredimss, output_coredimss
    )

    # Main code:
    ## Cast all input arrays to cubed
    # TODO there is no cubed.asarray yet
    args = [asarray(a) for a in args]

    if len(input_coredimss) != len(args):
        raise ValueError(
            f"According to `signature`, `func` requires {len(input_coredimss)} arguments, but {len(args)} given"
        )

    ## Axes: transpose input arguments
    transposed_args = []
    for arg, iax, input_coredims in zip(args, input_axes, input_coredimss):
        shape = arg.shape
        iax = tuple(a if a < 0 else a - len(shape) for a in iax)
        tidc = tuple(i for i in range(-len(shape) + 0, 0) if i not in iax) + iax
        transposed_arg = arg.transpose(tidc)
        transposed_args.append(transposed_arg)
    args = transposed_args

    ## Assess input args for loop dims
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
    input_dimss = [l + c for l, c in zip(loop_input_dimss, input_coredimss)]

    loop_output_dims = max(loop_input_dimss, key=len) if loop_input_dimss else tuple()

    ## Assess input args for same size and chunk sizes
    ### Collect sizes and chunksizes of all dims in all arrays
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
    ### Assert correct partitioning, for case:
    for dim, sizes in dimsizess.items():
        #### Check that the arrays have same length for same dimensions or dimension `1`
        if set(sizes) | {1} != {1, max(sizes)}:
            raise ValueError(f"Dimension `'{dim}'` with different lengths in arrays")

        # TODO do we actually want to allow other case of allow_rechunk == True?
        if not allow_rechunk:
            chunksizes = chunksizess[dim]
            #### Check if core dimensions consist of only one chunk
            if (dim in core_shapes) and (chunksizes[0][0] < core_shapes[dim]):
                raise ValueError(
                    "Core dimension `'{}'` consists of multiple chunks. To fix, rechunk into a single \
chunk along this dimension or set `allow_rechunk=True`, but beware that this may increase memory usage \
significantly.".format(
                        dim
                    )
                )
            #### Check if loop dimensions consist of same chunksizes, when they have sizes > 1
            relevant_chunksizes = list(
                unique(c for s, c in zip(sizes, chunksizes) if s > 1)
            )
            if len(relevant_chunksizes) > 1:
                raise ValueError(
                    f"Dimension `'{dim}'` with different chunksize present"
                )

    ## Apply function - use blockwise here
    arginds = list(concat(zip(args, input_dimss)))

    ### Use existing `blockwise` but only with loopdims to enforce
    ### concatenation for coredims that appear also at the output
    ### Modifying `blockwise` could improve things here.
    # TODO change meta
    # TODO cubed.blockwise needs a concatenate kwarg
    tmp = blockwise(
        func, loop_output_dims, *arginds, concatenate=True, meta=meta, **kwargs
    )

    # TODO more messing around with meta

    ## Prepare output shapes
    loop_output_shape = tmp.shape
    loop_output_chunks = tmp.chunks
    keys = list(flatten(tmp.__dask_keys__()))
    name, token = keys[0][0].split("-")

    ### *) Treat direct output
    if nout is None:
        output_coredimss = [output_coredimss]

    ## Split output
    leaf_arrs = []
    for i, (ocd, oax, meta) in enumerate(zip(output_coredimss, output_axes, metas)):
        core_output_shape = tuple(core_shapes[d] for d in ocd)
        core_chunkinds = len(ocd) * (0,)
        output_shape = loop_output_shape + core_output_shape
        output_chunks = loop_output_chunks + core_output_shape
        leaf_name = "%s_%d-%s" % (name, i, token)
        leaf_dsk = {
            (leaf_name,)
            + key[1:]
            + core_chunkinds: ((getitem, key, i) if nout else key)
            for key in keys
        }
        graph = HighLevelGraph.from_collections(leaf_name, leaf_dsk, dependencies=[tmp])
        meta = meta_from_array(meta, len(output_shape))
        leaf_arr = Array(
            graph, leaf_name, chunks=output_chunks, shape=output_shape, meta=meta
        )

        ### Axes:
        if keepdims:
            slices = len(leaf_arr.shape) * (slice(None),) + len(oax) * (np.newaxis,)
            leaf_arr = leaf_arr[slices]

        tidcs = [None] * len(leaf_arr.shape)
        for ii, oa in zip(range(-len(oax), 0), oax):
            tidcs[oa] = ii
        j = 0
        for ii in range(len(tidcs)):
            if tidcs[ii] is None:
                tidcs[ii] = j
                j += 1
        leaf_arr = leaf_arr.transpose(tidcs)
        leaf_arrs.append(leaf_arr)

    return (*leaf_arrs,) if nout else leaf_arrs[0]  # Undo *) from above
