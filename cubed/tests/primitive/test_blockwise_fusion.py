import collections.abc
from itertools import product
from typing import Iterator

from cubed.primitive.blockwise import (
    BlockwiseSpec,
    fuse_blockwise_specs,
    make_fused_key_function,
)
from cubed.utils import map_nested


def make_map_blocks_key_function(*names):
    def key_function(out_key):
        out_coords = out_key[1:]
        return tuple((name, *out_coords) for name in names)

    return key_function


def make_combine_blocks_list_key_function(name, numblocks, split_every):
    # similar to the key function in partial_reduce
    def key_function(out_key):
        out_coords = out_key[1:]

        # return a tuple with a single item that is a list of input keys to be combined
        in_keys = [
            list(
                range(
                    bi * split_every,
                    min((bi + 1) * split_every, numblocks),
                )
            )
            for bi in out_coords
        ]
        return ([(name,) + tuple(p) for p in product(*in_keys)],)

    return key_function


def make_combine_blocks_iter_key_function(name, numblocks, split_every):
    # similar to the key function in partial_reduce
    def key_function(out_key):
        out_coords = out_key[1:]

        # return a tuple with a single item that is an iterator of input keys to be combined
        in_keys = [
            list(
                range(
                    bi * split_every,
                    min((bi + 1) * split_every, numblocks),
                )
            )
            for bi in out_coords
        ]
        return (iter([(name,) + tuple(p) for p in product(*in_keys)]),)

    return key_function


def negative(x):
    assert isinstance(x, int)
    return -x


def add(x, y):
    assert isinstance(x, int)
    assert isinstance(y, int)
    return x + y


def sum_iter(x):
    assert isinstance(x, Iterator)
    return sum(a for a in x)


def check_key_function(key_function, out_coords, expected_str):
    res = key_function(("out",) + out_coords)

    assert str(tuple(iter_repr_nested(r) for r in res)) == expected_str


def iter_repr_nested(seq):
    # convert nested iterators to lists
    if isinstance(seq, list):
        return [iter_repr_nested(item) for item in seq]
    elif isinstance(seq, Iterator):
        return IteratorWithRepr([iter_repr_nested(item) for item in seq])
    else:
        return seq


class IteratorWithRepr(collections.abc.Iterator):
    def __init__(self, values):
        self.values = values
        self.it = iter(values)

    def __next__(self):
        return next(self.it)

    def __repr__(self):
        return "<" + ", ".join(repr(v) for v in self.values) + ">"

    def __str__(self):
        return "<" + ", ".join(str(v) for v in self.values) + ">"


def make_blockwise_spec(
    key_function,
    function,
    function_nargs=1,
    num_input_blocks=(1,),
    num_output_blocks=(1,),
    iterable_input_blocks=(False,),
):
    return BlockwiseSpec(
        key_function=key_function,
        function=function,
        function_nargs=function_nargs,
        num_input_blocks=num_input_blocks,
        num_output_blocks=num_output_blocks,
        iterable_input_blocks=iterable_input_blocks,
        reads_map={},  # unused
        writes_list=[],  # unused
    )


def test_map_blocks_key_function():
    key_function = make_map_blocks_key_function("a")

    check_key_function(key_function, (0,), "(('a', 0),)")
    check_key_function(key_function, (1,), "(('a', 1),)")


def test_map_blocks_multiple_inputs_key_function():
    key_function = make_map_blocks_key_function("a", "b")

    check_key_function(key_function, (0,), "(('a', 0), ('b', 0))")
    check_key_function(key_function, (1,), "(('a', 1), ('b', 1))")


def test_combine_blocks_list_key_function():
    key_function = make_combine_blocks_list_key_function(
        "a", numblocks=5, split_every=2
    )

    check_key_function(key_function, (0,), "([('a', 0), ('a', 1)],)")
    check_key_function(key_function, (1,), "([('a', 2), ('a', 3)],)")
    check_key_function(key_function, (2,), "([('a', 4)],)")


def test_combine_blocks_iter_key_function():
    key_function = make_combine_blocks_iter_key_function(
        "a", numblocks=5, split_every=2
    )

    check_key_function(key_function, (0,), "(<('a', 0), ('a', 1)>,)")
    check_key_function(key_function, (1,), "(<('a', 2), ('a', 3)>,)")
    check_key_function(key_function, (2,), "(<('a', 4)>,)")


def test_fuse_key_function_single_multiple():
    key_function1 = make_map_blocks_key_function("a")
    key_function2 = make_combine_blocks_iter_key_function(
        "b", numblocks=5, split_every=2
    )
    fused_key_function = make_fused_key_function(key_function2, [key_function1], [1])

    check_key_function(fused_key_function, (0,), "([<('a', 0), ('a', 1)>],)")
    check_key_function(fused_key_function, (1,), "([<('a', 2), ('a', 3)>],)")
    check_key_function(fused_key_function, (2,), "([<('a', 4)>],)")


def test_fuse_key_function_multiple_single():
    key_function1 = make_combine_blocks_iter_key_function(
        "a", numblocks=5, split_every=2
    )
    key_function2 = make_map_blocks_key_function("b")
    fused_key_function = make_fused_key_function(key_function2, [key_function1], [1])

    check_key_function(fused_key_function, (0,), "([<('a', 0), ('a', 1)>],)")
    check_key_function(fused_key_function, (1,), "([<('a', 2), ('a', 3)>],)")
    check_key_function(fused_key_function, (2,), "([<('a', 4)>],)")


def test_fuse_key_function_multiple_multiple():
    key_function1 = make_combine_blocks_iter_key_function(
        "a", numblocks=5, split_every=2
    )
    key_function2 = make_combine_blocks_iter_key_function(
        "b", numblocks=3, split_every=2
    )
    fused_key_function = make_fused_key_function(key_function2, [key_function1], [1])

    check_key_function(
        fused_key_function, (0,), "([<<('a', 0), ('a', 1)>, <('a', 2), ('a', 3)>>],)"
    )
    check_key_function(fused_key_function, (1,), "([<<('a', 4)>>],)")


def apply_blockwise(input_data, out_coords, bw_spec):
    args = []
    out_key = ("out",) + tuple(out_coords)  # array name is ignored by key_function
    in_keys = bw_spec.key_function(out_key)
    for in_key in in_keys:
        # just return the (1D) coord as a value
        def get_data(key):
            name = key[0]
            index = key[1]  # 1d index
            return input_data[name][index]

        arg = map_nested(get_data, in_key)
        args.append(arg)
    return bw_spec.function(*args)


def test_apply_blockwise():
    bw_spec = make_blockwise_spec(
        key_function=make_map_blocks_key_function("a"),
        function=negative,
    )

    input_data = {"a": [0, 1, 2, 3, 4]}
    out = [apply_blockwise(input_data, [i], bw_spec) for i in range(5)]
    assert out == [0, -1, -2, -3, -4]


def test_apply_blockwise_multiple_inputs():
    bw_spec = make_blockwise_spec(
        key_function=make_map_blocks_key_function("a", "b"), function=add
    )

    input_data = {"a": [0, 1, 2, 3, 4], "b": [5, 6, 7, 8, 9]}
    out = [apply_blockwise(input_data, [i], bw_spec) for i in range(5)]
    assert out == [5, 7, 9, 11, 13]


def test_apply_blockwise_iterator():
    bw_spec = make_blockwise_spec(
        key_function=make_combine_blocks_iter_key_function(
            "a", numblocks=5, split_every=2
        ),
        function=sum_iter,
    )

    input_data = {"a": [0, 1, 2, 3, 4]}
    out = [apply_blockwise(input_data, [i], bw_spec) for i in range(3)]
    assert out == [1, 5, 4]


def test_apply_blockwise_fused():
    bw_spec1 = make_blockwise_spec(
        key_function=make_map_blocks_key_function("a"), function=negative
    )
    bw_spec2 = make_blockwise_spec(
        key_function=make_map_blocks_key_function("b"), function=negative
    )

    bw_spec = fuse_blockwise_specs(bw_spec2, bw_spec1)

    input_data = {"a": [0, 1, 2, 3, 4]}
    out = [apply_blockwise(input_data, [i], bw_spec) for i in range(5)]
    assert out == [0, 1, 2, 3, 4]


def test_apply_blockwise_fused_iterator():
    bw_spec1 = make_blockwise_spec(
        key_function=make_map_blocks_key_function("a"), function=negative
    )
    bw_spec2 = make_blockwise_spec(
        key_function=make_combine_blocks_iter_key_function(
            "b", numblocks=5, split_every=2
        ),
        function=sum_iter,
        num_input_blocks=(2,),
        iterable_input_blocks=(True,),
    )

    bw_spec = fuse_blockwise_specs(bw_spec2, bw_spec1)

    input_data = {"a": [0, 1, 2, 3, 4]}
    out = [apply_blockwise(input_data, [i], bw_spec) for i in range(3)]
    assert out == [-1, -5, -4]


def test_apply_blockwise_fused_iterator_with_single_input_block():
    bw_spec1 = make_blockwise_spec(
        key_function=make_map_blocks_key_function("a"), function=negative
    )
    # the iterator only reads from a single input block
    bw_spec2 = make_blockwise_spec(
        key_function=make_combine_blocks_iter_key_function(
            "b", numblocks=5, split_every=1
        ),
        function=sum_iter,
        num_input_blocks=(1,),
        iterable_input_blocks=(True,),
    )

    bw_spec = fuse_blockwise_specs(bw_spec2, bw_spec1)

    input_data = {"a": [0, 1, 2, 3, 4]}
    out = [apply_blockwise(input_data, [i], bw_spec) for i in range(5)]
    assert out == [0, -1, -2, -3, -4]
