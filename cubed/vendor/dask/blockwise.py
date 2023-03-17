from __future__ import annotations

import itertools

import tlz as toolz

from .core import flatten


def _get_coord_mapping(
    dims,
    output,
    out_indices,
    numblocks,
    argpairs,
    concatenate,
):
    """Calculate coordinate mapping for graph construction.

    This function handles the high-level logic behind Blockwise graph
    construction. The output is a tuple containing: The mapping between
    input and output block coordinates (`coord_maps`), the axes along
    which to concatenate for each input (`concat_axes`), and the dummy
    indices needed for broadcasting (`dummies`).

    Used by `make_blockwise_graph` and `Blockwise._cull_dependencies`.

    Parameters
    ----------
    dims : dict
        Mapping between each index specified in `argpairs` and
        the number of output blocks for that index. Corresponds
        to the Blockwise `dims` attribute.
    output : str
        Corresponds to the Blockwise `output` attribute.
    out_indices : tuple
        Corresponds to the Blockwise `output_indices` attribute.
    numblocks : dict
        Corresponds to the Blockwise `numblocks` attribute.
    argpairs : tuple
        Corresponds to the Blockwise `indices` attribute.
    concatenate : bool
        Corresponds to the Blockwise `concatenate` attribute.
    """

    block_names = set()
    all_indices = set()
    for name, ind in argpairs:
        if ind is not None:
            block_names.add(name)
            for x in ind:
                all_indices.add(x)
    assert set(numblocks) == block_names

    dummy_indices = all_indices - set(out_indices)

    # For each position in the output space, we'll construct a
    # "coordinate set" that consists of
    # - the output indices
    # - the dummy indices
    # - the dummy indices, with indices replaced by zeros (for broadcasting), we
    #   are careful to only emit a single dummy zero when concatenate=True to not
    #   concatenate the same array with itself several times.
    # - a 0 to assist with broadcasting.

    index_pos, zero_pos = {}, {}
    for i, ind in enumerate(out_indices):
        index_pos[ind] = i
        zero_pos[ind] = -1

    _dummies_list = []
    for i, ind in enumerate(dummy_indices):
        index_pos[ind] = 2 * i + len(out_indices)
        zero_pos[ind] = 2 * i + 1 + len(out_indices)
        reps = 1 if concatenate else dims[ind]
        _dummies_list.append([list(range(dims[ind])), [0] * reps])

    # ([0, 1, 2], [0, 0, 0], ...)  For a dummy index of dimension 3
    dummies = tuple(itertools.chain.from_iterable(_dummies_list))
    dummies += (0,)

    # For each coordinate position in each input, gives the position in
    # the coordinate set.
    coord_maps = []

    # Axes along which to concatenate, for each input
    concat_axes = []
    for arg, ind in argpairs:
        if ind is not None:
            coord_maps.append(
                [
                    zero_pos[i] if nb == 1 else index_pos[i]
                    for i, nb in zip(ind, numblocks[arg])
                ]
            )
            concat_axes.append([n for n, i in enumerate(ind) if i in dummy_indices])
        else:
            coord_maps.append(None)
            concat_axes.append(None)

    return coord_maps, concat_axes, dummies


def make_blockwise_graph(
    func,
    output,
    out_indices,
    *arrind_pairs,
    numblocks=None,
    concatenate=None,
    new_axes=None,
    output_blocks=None,
    dims=None,
    deserializing=False,
    func_future_args=None,
    return_key_deps=False,
    io_deps=None,
):
    """Tensor operation

    Applies a function, ``func``, across blocks from many different input
    collections.  We arrange the pattern with which those blocks interact with
    sets of matching indices.  E.g.::

        make_blockwise_graph(func, 'z', 'i', 'x', 'i', 'y', 'i')

    yield an embarrassingly parallel communication pattern and is read as

        $$ z_i = func(x_i, y_i) $$

    More complex patterns may emerge, including multiple indices::

        make_blockwise_graph(func, 'z', 'ij', 'x', 'ij', 'y', 'ji')

        $$ z_{ij} = func(x_{ij}, y_{ji}) $$

    Indices missing in the output but present in the inputs results in many
    inputs being sent to one function (see examples).

    Examples
    --------
    Simple embarrassing map operation

    >>> inc = lambda x: x + 1
    >>> make_blockwise_graph(inc, 'z', 'ij', 'x', 'ij', numblocks={'x': (2, 2)})  # doctest: +SKIP
    {('z', 0, 0): (inc, ('x', 0, 0)),
     ('z', 0, 1): (inc, ('x', 0, 1)),
     ('z', 1, 0): (inc, ('x', 1, 0)),
     ('z', 1, 1): (inc, ('x', 1, 1))}

    Simple operation on two datasets

    >>> add = lambda x, y: x + y
    >>> make_blockwise_graph(add, 'z', 'ij', 'x', 'ij', 'y', 'ij', numblocks={'x': (2, 2),
    ...                                                      'y': (2, 2)})  # doctest: +SKIP
    {('z', 0, 0): (add, ('x', 0, 0), ('y', 0, 0)),
     ('z', 0, 1): (add, ('x', 0, 1), ('y', 0, 1)),
     ('z', 1, 0): (add, ('x', 1, 0), ('y', 1, 0)),
     ('z', 1, 1): (add, ('x', 1, 1), ('y', 1, 1))}

    Operation that flips one of the datasets

    >>> addT = lambda x, y: x + y.T  # Transpose each chunk
    >>> #                                        z_ij ~ x_ij y_ji
    >>> #               ..         ..         .. notice swap
    >>> make_blockwise_graph(addT, 'z', 'ij', 'x', 'ij', 'y', 'ji', numblocks={'x': (2, 2),
    ...                                                       'y': (2, 2)})  # doctest: +SKIP
    {('z', 0, 0): (add, ('x', 0, 0), ('y', 0, 0)),
     ('z', 0, 1): (add, ('x', 0, 1), ('y', 1, 0)),
     ('z', 1, 0): (add, ('x', 1, 0), ('y', 0, 1)),
     ('z', 1, 1): (add, ('x', 1, 1), ('y', 1, 1))}

    Dot product with contraction over ``j`` index.  Yields list arguments

    >>> make_blockwise_graph(dotmany, 'z', 'ik', 'x', 'ij', 'y', 'jk', numblocks={'x': (2, 2),
    ...                                                          'y': (2, 2)})  # doctest: +SKIP
    {('z', 0, 0): (dotmany, [('x', 0, 0), ('x', 0, 1)],
                            [('y', 0, 0), ('y', 1, 0)]),
     ('z', 0, 1): (dotmany, [('x', 0, 0), ('x', 0, 1)],
                            [('y', 0, 1), ('y', 1, 1)]),
     ('z', 1, 0): (dotmany, [('x', 1, 0), ('x', 1, 1)],
                            [('y', 0, 0), ('y', 1, 0)]),
     ('z', 1, 1): (dotmany, [('x', 1, 0), ('x', 1, 1)],
                            [('y', 0, 1), ('y', 1, 1)])}

    Pass ``concatenate=True`` to concatenate arrays ahead of time

    >>> make_blockwise_graph(f, 'z', 'i', 'x', 'ij', 'y', 'ij', concatenate=True,
    ...     numblocks={'x': (2, 2), 'y': (2, 2,)})  # doctest: +SKIP
    {('z', 0): (f, (concatenate_axes, [('x', 0, 0), ('x', 0, 1)], (1,)),
                   (concatenate_axes, [('y', 0, 0), ('y', 0, 1)], (1,)))
     ('z', 1): (f, (concatenate_axes, [('x', 1, 0), ('x', 1, 1)], (1,)),
                   (concatenate_axes, [('y', 1, 0), ('y', 1, 1)], (1,)))}

    Supports Broadcasting rules

    >>> make_blockwise_graph(add, 'z', 'ij', 'x', 'ij', 'y', 'ij', numblocks={'x': (1, 2),
    ...                                                      'y': (2, 2)})  # doctest: +SKIP
    {('z', 0, 0): (add, ('x', 0, 0), ('y', 0, 0)),
     ('z', 0, 1): (add, ('x', 0, 1), ('y', 0, 1)),
     ('z', 1, 0): (add, ('x', 0, 0), ('y', 1, 0)),
     ('z', 1, 1): (add, ('x', 0, 1), ('y', 1, 1))}

    Support keyword arguments with apply

    >>> def f(a, b=0): return a + b
    >>> make_blockwise_graph(f, 'z', 'i', 'x', 'i', numblocks={'x': (2,)}, b=10)  # doctest: +SKIP
    {('z', 0): (apply, f, [('x', 0)], {'b': 10}),
     ('z', 1): (apply, f, [('x', 1)], {'b': 10})}

    Include literals by indexing with ``None``

    >>> make_blockwise_graph(add, 'z', 'i', 'x', 'i', 100, None,  numblocks={'x': (2,)})  # doctest: +SKIP
    {('z', 0): (add, ('x', 0), 100),
     ('z', 1): (add, ('x', 1), 100)}

    See Also
    --------
    dask.array.blockwise
    dask.blockwise.blockwise
    """

    if numblocks is None:
        raise ValueError("Missing required numblocks argument.")
    new_axes = new_axes or {}
    io_deps = io_deps or {}
    argpairs = list(toolz.partition(2, arrind_pairs))

    if return_key_deps:
        key_deps = {}

    if deserializing:
        from dask.utils import stringify_collection_keys
        from distributed.protocol.serialize import to_serialize

    if concatenate is True:
        from dask.array.core import concatenate_axes as concatenate

    # Dictionary mapping {i: 3, j: 4, ...} for i, j, ... the dimensions
    dims = dims or _make_dims(argpairs, numblocks, new_axes)

    # Generate the abstract "plan" before constructing
    # the actual graph
    (coord_maps, concat_axes, dummies) = _get_coord_mapping(
        dims,
        output,
        out_indices,
        numblocks,
        argpairs,
        concatenate,
    )

    # Apply Culling.
    # Only need to construct the specified set of output blocks.
    # Note that we must convert itertools.product to list,
    # because we may need to loop through output_blocks more than
    # once below (itertools.product already uses an internal list,
    # so this is not a memory regression)
    output_blocks = output_blocks or list(
        itertools.product(*[range(dims[i]) for i in out_indices])
    )

    dsk = {}
    # Create argument lists
    for out_coords in output_blocks:
        deps = set()
        coords = out_coords + dummies
        args = []
        for cmap, axes, (arg, ind) in zip(coord_maps, concat_axes, argpairs):
            if ind is None:
                if deserializing:
                    args.append(stringify_collection_keys(arg))
                else:
                    args.append(arg)
            else:
                arg_coords = tuple(coords[c] for c in cmap)
                if axes:
                    tups = lol_product((arg,), arg_coords)
                    if arg not in io_deps:
                        deps.update(flatten(tups))

                    if concatenate:
                        tups = (concatenate, tups, axes)
                else:
                    tups = (arg,) + arg_coords
                    if arg not in io_deps:
                        deps.add(tups)
                # Replace "place-holder" IO keys with "real" args
                if arg in io_deps:
                    # We don't want to stringify keys for args
                    # we are replacing here
                    idx = tups[1:]
                    args.append(io_deps[arg].get(idx, idx))
                elif deserializing:
                    args.append(stringify_collection_keys(tups))
                else:
                    args.append(tups)
        out_key = (output,) + out_coords

        if deserializing:
            deps.update(func_future_args)
            args += list(func_future_args)

        # Construct a function/args/kwargs dict if we
        # do not have a nested task (i.e. concatenate=False).
        # TODO: Avoid using the iterate_collection-version
        # of to_serialize if we know that are no embedded
        # Serialized/Serialize objects in args and/or kwargs.
        if deserializing and isinstance(func, bytes):
            dsk[out_key] = {"function": func, "args": to_serialize(args)}
        else:
            args.insert(0, func)
            val = tuple(args)
            # May still need to serialize (if concatenate=True)
            dsk[out_key] = to_serialize(val) if deserializing else val

        if return_key_deps:
            key_deps[out_key] = deps

    if return_key_deps:
        # Add valid-key dependencies from io_deps
        for key, io_dep in io_deps.items():
            if io_dep.produces_keys:
                for out_coords in output_blocks:
                    key = (output,) + out_coords
                    valid_key_dep = io_dep[out_coords]
                    key_deps[key] |= {valid_key_dep}

        return dsk, key_deps
    else:
        return dsk


def lol_product(head, values):
    """List of list of tuple keys, similar to `itertools.product`.

    Parameters
    ----------
    head : tuple
        Prefix prepended to all results.
    values : sequence
        Mix of singletons and lists. Each list is substituted with every
        possible value and introduces another level of list in the output.

    Examples
    --------
    >>> lol_product(('x',), (1, 2, 3))
    ('x', 1, 2, 3)
    >>> lol_product(('x',), (1, [2, 3], 4, [5, 6]))  # doctest: +NORMALIZE_WHITESPACE
    [[('x', 1, 2, 4, 5), ('x', 1, 2, 4, 6)],
     [('x', 1, 3, 4, 5), ('x', 1, 3, 4, 6)]]
    """
    if not values:
        return head
    elif isinstance(values[0], list):
        return [lol_product(head + (x,), values[1:]) for x in values[0]]
    else:
        return lol_product(head + (values[0],), values[1:])







def broadcast_dimensions(argpairs, numblocks, sentinels=(1, (1,)), consolidate=None):
    """Find block dimensions from arguments

    Parameters
    ----------
    argpairs : iterable
        name, ijk index pairs
    numblocks : dict
        maps {name: number of blocks}
    sentinels : iterable (optional)
        values for singleton dimensions
    consolidate : func (optional)
        use this to reduce each set of common blocks into a smaller set

    Examples
    --------
    >>> argpairs = [('x', 'ij'), ('y', 'ji')]
    >>> numblocks = {'x': (2, 3), 'y': (3, 2)}
    >>> broadcast_dimensions(argpairs, numblocks)
    {'i': 2, 'j': 3}

    Supports numpy broadcasting rules

    >>> argpairs = [('x', 'ij'), ('y', 'ij')]
    >>> numblocks = {'x': (2, 1), 'y': (1, 3)}
    >>> broadcast_dimensions(argpairs, numblocks)
    {'i': 2, 'j': 3}

    Works in other contexts too

    >>> argpairs = [('x', 'ij'), ('y', 'ij')]
    >>> d = {'x': ('Hello', 1), 'y': (1, (2, 3))}
    >>> broadcast_dimensions(argpairs, d)
    {'i': 'Hello', 'j': (2, 3)}
    """
    # List like [('i', 2), ('j', 1), ('i', 1), ('j', 2)]
    argpairs2 = [(a, ind) for a, ind in argpairs if ind is not None]
    L = toolz.concat(
        [
            zip(inds, dims)
            for (x, inds), (x, dims) in toolz.join(
                toolz.first, argpairs2, toolz.first, numblocks.items()
            )
        ]
    )

    g = toolz.groupby(0, L)
    g = {k: {d for i, d in v} for k, v in g.items()}

    g2 = {k: v - set(sentinels) if len(v) > 1 else v for k, v in g.items()}

    if consolidate:
        return toolz.valmap(consolidate, g2)

    if g2 and not set(map(len, g2.values())) == {1}:
        raise ValueError("Shapes do not align %s" % g)

    return toolz.valmap(toolz.first, g2)


def _make_dims(indices, numblocks, new_axes):
    """Returns a dictionary mapping between each index specified in
    `indices` and the number of output blocks for that indice.
    """
    dims = broadcast_dimensions(indices, numblocks)
    for k, v in new_axes.items():
        dims[k] = len(v) if isinstance(v, tuple) else 1
    return dims
