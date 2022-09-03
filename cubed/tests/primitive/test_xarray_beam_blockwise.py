import pytest
from dask.blockwise import make_blockwise_graph
from xarray_beam import Key

from cubed.primitive.xarray_beam_blockwise import (
    get_index_to_offset,
    get_offset_to_index,
    index_to_xbeam_key,
    invert_blockwise_graph,
    make_inverse_blockwise_function_dynamic,
    xbeam_key_to_index,
    xbeam_key_wrapper,
)


def test_make_inverse_blockwise_function_dynamic_map():
    block_fn = make_inverse_blockwise_function_dynamic(
        "z", "ij", "x", "ij", numblocks={"x": (2, 2)}
    )

    assert block_fn("x", (0, 0)) == [(0, 0)]
    assert block_fn("x", (0, 1)) == [(0, 1)]
    assert block_fn("x", (1, 0)) == [(1, 0)]
    assert block_fn("x", (1, 1)) == [(1, 1)]

    graph = make_blockwise_graph(
        lambda x: 0, "z", "ij", "x", "ij", numblocks={"x": (2, 2)}
    )
    check_consistent_with_graph(block_fn, graph, ("x",))


def test_make_inverse_blockwise_function_dynamic_elemwise():
    block_fn = make_inverse_blockwise_function_dynamic(
        "z", "ij", "x", "ij", "y", "ij", numblocks={"x": (2, 2), "y": (2, 2)}
    )

    assert block_fn("x", (0, 0)) == [(0, 0)]
    assert block_fn("x", (0, 1)) == [(0, 1)]
    assert block_fn("x", (1, 0)) == [(1, 0)]
    assert block_fn("x", (1, 1)) == [(1, 1)]

    assert block_fn("y", (0, 0)) == [(0, 0)]
    assert block_fn("y", (0, 1)) == [(0, 1)]
    assert block_fn("y", (1, 0)) == [(1, 0)]
    assert block_fn("y", (1, 1)) == [(1, 1)]

    graph = make_blockwise_graph(
        lambda x, y: 0,
        "z",
        "ij",
        "x",
        "ij",
        "y",
        "ij",
        numblocks={"x": (2, 2), "y": (2, 2)},
    )
    check_consistent_with_graph(block_fn, graph, ("x", "y"))


def test_make_inverse_blockwise_function_dynamic_flip():
    block_fn = make_inverse_blockwise_function_dynamic(
        "z", "ij", "x", "ij", "y", "ji", numblocks={"x": (2, 2), "y": (2, 2)}
    )

    assert block_fn("x", (0, 0)) == [(0, 0)]
    assert block_fn("x", (0, 1)) == [(0, 1)]
    assert block_fn("x", (1, 0)) == [(1, 0)]
    assert block_fn("x", (1, 1)) == [(1, 1)]

    assert block_fn("y", (0, 0)) == [(0, 0)]
    assert block_fn("y", (0, 1)) == [(1, 0)]
    assert block_fn("y", (1, 0)) == [(0, 1)]
    assert block_fn("y", (1, 1)) == [(1, 1)]

    graph = make_blockwise_graph(
        lambda x, y: 0,
        "z",
        "ij",
        "x",
        "ij",
        "y",
        "ji",
        numblocks={"x": (2, 2), "y": (2, 2)},
    )
    check_consistent_with_graph(block_fn, graph, ("x", "y"))


def test_make_inverse_blockwise_function_dynamic_contract():
    block_fn = make_inverse_blockwise_function_dynamic(
        "z", "ik", "x", "ij", "y", "jk", numblocks={"x": (2, 2), "y": (2, 2)}
    )

    assert block_fn("x", (0, 0)) == [(0, 0), (0, 1)]
    assert block_fn("x", (0, 1)) == [(0, 0), (0, 1)]
    assert block_fn("x", (1, 0)) == [(1, 0), (1, 1)]
    assert block_fn("x", (1, 1)) == [(1, 0), (1, 1)]

    assert block_fn("y", (0, 0)) == [(0, 0), (1, 0)]
    assert block_fn("y", (0, 1)) == [(0, 1), (1, 1)]
    assert block_fn("y", (1, 0)) == [(0, 0), (1, 0)]
    assert block_fn("y", (1, 1)) == [(0, 1), (1, 1)]

    graph = make_blockwise_graph(
        lambda x, y: 0,
        "z",
        "ik",
        "x",
        "ij",
        "y",
        "jk",
        numblocks={"x": (2, 2), "y": (2, 2)},
    )
    check_consistent_with_graph(block_fn, graph, ("x", "y"))


@pytest.mark.xfail
def test_make_inverse_blockwise_function_dynamic_contract_1d():
    block_fn = make_inverse_blockwise_function_dynamic(
        "z", "j", "x", "ij", numblocks={"x": (1, 2)}
    )

    assert block_fn("x", (0, 0)) == [(0,)]
    assert block_fn("x", (0, 1)) == [(1,)]

    graph = make_blockwise_graph(
        lambda x: 0, "z", "j", "x", "ij", numblocks={"x": (1, 2)}
    )
    check_consistent_with_graph(block_fn, graph, ("x",))


@pytest.mark.xfail
def test_make_inverse_blockwise_function_dynamic_contract_0d():
    block_fn = make_inverse_blockwise_function_dynamic(
        "z", "", "x", "ij", numblocks={"x": (2, 2)}
    )

    assert block_fn("x", (0, 0)) == [()]
    assert block_fn("x", (0, 1)) == [()]
    assert block_fn("x", (1, 0)) == [()]
    assert block_fn("x", (1, 1)) == [()]

    graph = make_blockwise_graph(
        lambda x: 0, "z", "", "x", "ij", numblocks={"x": (2, 2)}
    )
    check_consistent_with_graph(block_fn, graph, ("x",))


def test_make_inverse_blockwise_function_dynamic_broadcast():
    block_fn = make_inverse_blockwise_function_dynamic(
        "z", "ij", "x", "ij", "y", "ij", numblocks={"x": (1, 2), "y": (2, 2)}
    )

    assert block_fn("x", (0, 0)) == [(0, 0), (1, 0)]
    assert block_fn("x", (0, 1)) == [(0, 1), (1, 1)]

    assert block_fn("y", (0, 0)) == [(0, 0)]
    assert block_fn("y", (0, 1)) == [(0, 1)]
    assert block_fn("y", (1, 0)) == [(1, 0)]
    assert block_fn("y", (1, 1)) == [(1, 1)]

    graph = make_blockwise_graph(
        lambda x, y: 0,
        "z",
        "ij",
        "x",
        "ij",
        "y",
        "ij",
        numblocks={"x": (1, 2), "y": (2, 2)},
    )
    check_consistent_with_graph(block_fn, graph, ("x", "y"))


def check_consistent_with_graph(block_fn, graph, args):
    inverses = invert_blockwise_graph(graph)

    for arg in args:
        for k, vs in inverses[arg].items():
            assert vs == block_fn(arg, k)


def test_xbeam_key_wrapper_same_shapes_and_chunks():
    block_fn = make_inverse_blockwise_function_dynamic(
        "z", "ij", "x", "ij", "y", "ij", numblocks={"x": (2, 2), "y": (2, 2)}
    )

    in_shape_map = {
        "x": (10, 8),
        "y": (10, 8),
    }

    in_chunks_map = {
        "x": (5, 4),
        "y": (5, 4),
    }

    out_shape = (10, 8)
    out_chunks = (5, 4)

    xbeam_block_func = xbeam_key_wrapper(
        block_fn, in_shape_map, in_chunks_map, out_shape, out_chunks
    )

    assert xbeam_block_func("x", Key(offsets={"dim_0": 0, "dim_1": 0})) == [
        Key(offsets={"dim_0": 0, "dim_1": 0})
    ]
    assert xbeam_block_func("x", Key(offsets={"dim_0": 0, "dim_1": 4})) == [
        Key(offsets={"dim_0": 0, "dim_1": 4})
    ]
    assert xbeam_block_func("x", Key(offsets={"dim_0": 5, "dim_1": 0})) == [
        Key(offsets={"dim_0": 5, "dim_1": 0})
    ]
    assert xbeam_block_func("x", Key(offsets={"dim_0": 5, "dim_1": 4})) == [
        Key(offsets={"dim_0": 5, "dim_1": 4})
    ]

    assert xbeam_block_func("y", Key(offsets={"dim_0": 0, "dim_1": 0})) == [
        Key(offsets={"dim_0": 0, "dim_1": 0})
    ]
    assert xbeam_block_func("y", Key(offsets={"dim_0": 0, "dim_1": 4})) == [
        Key(offsets={"dim_0": 0, "dim_1": 4})
    ]
    assert xbeam_block_func("y", Key(offsets={"dim_0": 5, "dim_1": 0})) == [
        Key(offsets={"dim_0": 5, "dim_1": 0})
    ]
    assert xbeam_block_func("y", Key(offsets={"dim_0": 5, "dim_1": 4})) == [
        Key(offsets={"dim_0": 5, "dim_1": 4})
    ]


def test_xbeam_key_wrapper_different_shapes_and_chunks():
    block_fn = make_inverse_blockwise_function_dynamic(
        "z", "ij", "x", "ij", "y", "ij", numblocks={"x": (2, 2), "y": (2, 2)}
    )

    # Note that arrays have different shapes and chunk sizes, but numblocks
    # is the same for each

    in_shape_map = {
        "x": (2, 2),
        "y": (7, 7),
    }

    in_chunks_map = {
        "x": (1, 1),
        "y": (5, 5),
    }

    out_shape = (10, 8)
    out_chunks = (5, 4)

    xbeam_block_func = xbeam_key_wrapper(
        block_fn, in_shape_map, in_chunks_map, out_shape, out_chunks
    )

    assert xbeam_block_func("x", Key(offsets={"dim_0": 0, "dim_1": 0})) == [
        Key(offsets={"dim_0": 0, "dim_1": 0})
    ]
    assert xbeam_block_func("x", Key(offsets={"dim_0": 0, "dim_1": 1})) == [
        Key(offsets={"dim_0": 0, "dim_1": 4})
    ]
    assert xbeam_block_func("x", Key(offsets={"dim_0": 1, "dim_1": 0})) == [
        Key(offsets={"dim_0": 5, "dim_1": 0})
    ]
    assert xbeam_block_func("x", Key(offsets={"dim_0": 1, "dim_1": 1})) == [
        Key(offsets={"dim_0": 5, "dim_1": 4})
    ]

    assert xbeam_block_func("y", Key(offsets={"dim_0": 0, "dim_1": 0})) == [
        Key(offsets={"dim_0": 0, "dim_1": 0})
    ]
    assert xbeam_block_func("y", Key(offsets={"dim_0": 0, "dim_1": 5})) == [
        Key(offsets={"dim_0": 0, "dim_1": 4})
    ]
    assert xbeam_block_func("y", Key(offsets={"dim_0": 5, "dim_1": 0})) == [
        Key(offsets={"dim_0": 5, "dim_1": 0})
    ]
    assert xbeam_block_func("y", Key(offsets={"dim_0": 5, "dim_1": 5})) == [
        Key(offsets={"dim_0": 5, "dim_1": 4})
    ]


def test_index_to_xbeam_key():
    ito = get_index_to_offset(shape=(10, 8), chunks=(4, 3))

    assert index_to_xbeam_key((0, 0), ito) == Key(offsets={"dim_0": 0, "dim_1": 0})
    assert index_to_xbeam_key((0, 1), ito) == Key(offsets={"dim_0": 0, "dim_1": 3})
    assert index_to_xbeam_key((1, 0), ito) == Key(offsets={"dim_0": 4, "dim_1": 0})
    assert index_to_xbeam_key((2, 2), ito) == Key(offsets={"dim_0": 8, "dim_1": 6})


def test_xbeam_key_to_index():
    oti = get_offset_to_index(shape=(10, 8), chunks=(4, 3))

    assert xbeam_key_to_index(Key(offsets={"dim_0": 0, "dim_1": 0}), oti) == (0, 0)
    assert xbeam_key_to_index(Key(offsets={"dim_0": 0, "dim_1": 3}), oti) == (0, 1)
    assert xbeam_key_to_index(Key(offsets={"dim_0": 4, "dim_1": 0}), oti) == (1, 0)
    assert xbeam_key_to_index(Key(offsets={"dim_0": 8, "dim_1": 6}), oti) == (2, 2)
