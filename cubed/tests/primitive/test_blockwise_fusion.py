import collections.abc
import logging
import math
from collections.abc import Callable, Iterable, Iterator
from itertools import product
from typing import Any, TypeVar

from cubed.primitive.blockwise import (
    BlockwiseSpec,
    ChunkKey,
    FunctionArgs,
    KeyFunction,
    fuse_blockwise_specs,
    make_fused_back_key_function,
    map_nested,
)

logger = logging.getLogger(__name__)


def make_map_blocks_back_key_function(
    *names: str,
) -> Callable[[ChunkKey], FunctionArgs[ChunkKey]]:
    def back_key_function(out_key: ChunkKey) -> FunctionArgs[ChunkKey]:
        out_coords = out_key.coords
        return FunctionArgs(
            *tuple(ChunkKey(name, out_coords) for name in names),
            output_name=out_key.name,
        )

    return back_key_function


def make_combine_blocks_list_back_key_function(
    name: str, numblocks: int, split_every: int
) -> Callable[[ChunkKey], FunctionArgs[list[ChunkKey]]]:
    # similar to the key function in partial_reduce
    def back_key_function(out_key: ChunkKey) -> FunctionArgs[list[ChunkKey]]:
        out_coords = out_key.coords

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
        return FunctionArgs(
            [ChunkKey(name, tuple(p)) for p in product(*in_keys)],
            output_name=out_key.name,
        )

    return back_key_function


def make_combine_blocks_iter_back_key_function(
    name: str, numblocks: int, split_every: int
) -> Callable[[ChunkKey], FunctionArgs[Iterator[ChunkKey]]]:
    # similar to the key function in partial_reduce
    def back_key_function(out_key: ChunkKey) -> FunctionArgs[Iterator[ChunkKey]]:
        out_coords = out_key.coords

        # return a tuple with a single item that is an iterator of input keys
        # to be combined
        in_keys = [
            list(
                range(
                    bi * split_every,
                    min((bi + 1) * split_every, numblocks),
                )
            )
            for bi in out_coords
        ]
        return FunctionArgs(
            iter([ChunkKey(name, tuple(p)) for p in product(*in_keys)]),
            output_name=out_key.name,
        )

    return back_key_function


def make_alternate_blocks_back_key_function(
    name1: str, name2: str
) -> Callable[[ChunkKey], FunctionArgs[ChunkKey]]:
    # similar to the key function for stack
    def back_key_function(out_key: ChunkKey) -> FunctionArgs[ChunkKey]:
        out_coords = out_key.coords
        index = out_coords[0]  # 1d index
        name = name1 if index % 2 == 0 else name2
        return FunctionArgs(ChunkKey(name, out_coords), output_name=out_key.name)

    return back_key_function


def make_concat_blocks_back_key_function(
    name1: str, name2: str
) -> Callable[[ChunkKey], FunctionArgs[Iterator[ChunkKey]]]:
    # similar to the key function for concat
    def back_key_function(out_key: ChunkKey) -> FunctionArgs[Iterator[ChunkKey]]:
        out_coords = out_key.coords
        index = out_coords[0]  # 1d index
        if index == 0:
            it = iter([ChunkKey(name1, (0,))])
        elif index == 1:
            it = iter([ChunkKey(name1, (1,)), ChunkKey(name2, (0,))])
        elif index == 2:
            it = iter([ChunkKey(name2, (0,)), ChunkKey(name2, (1,))])
        elif index > 2:
            raise IndexError()
        return FunctionArgs(it, output_name=out_key.name)

    return back_key_function


def identity(x: int) -> int:
    logger.debug(f"identity({x})")
    return x


def negative(x: int) -> int:
    logger.debug(f"negative({x})")
    assert isinstance(x, int)
    return -x


def add(x: int, y: int) -> int:
    logger.debug(f"add({x}, {y})")
    assert isinstance(x, int)
    assert isinstance(y, int)
    return x + y


def sum_iter(x: Iterator[int]) -> int:
    logger.debug(f"sum_iter({x})")
    assert isinstance(x, Iterator)
    return sum(a for a in x)


def sum_list(x: list[int]) -> int:
    logger.debug(f"sum_list({x})")
    assert isinstance(x, list)
    return sum(a for a in x)


def int_sqrts(x: int) -> Iterator[int]:
    logger.debug(f"int_sqrts({x})")
    assert isinstance(x, int)
    s = int(math.sqrt(x))
    yield s
    yield -s


def check_back_key_function(
    back_key_function: KeyFunction,
    out_coords: tuple[int, ...],
    expected_str: str,
) -> None:
    res = back_key_function(ChunkKey("out", out_coords))
    assert isinstance(res, FunctionArgs)

    assert str(iter_repr_nested(res)) == expected_str


def iter_repr_nested(seq: Any) -> Any:
    # convert nested iterators to lists
    if isinstance(seq, list):
        return [iter_repr_nested(item) for item in seq]
    elif isinstance(seq, FunctionArgs):
        return FunctionArgs(
            *[iter_repr_nested(item) for item in seq.args], output_name=seq.output_name
        )
    elif isinstance(seq, Iterator):
        return IteratorWithRepr([iter_repr_nested(item) for item in seq])
    else:
        return seq


T = TypeVar("T")


class IteratorWithRepr(collections.abc.Iterator[T]):
    def __init__(self, values: Iterable[T]):
        self.values = values
        self.it = iter(values)

    def __next__(self) -> T:
        return next(self.it)

    def __repr__(self) -> str:
        return "<" + ", ".join(repr(v) for v in self.values) + ">"

    def __str__(self) -> str:
        return "<" + ", ".join(str(v) for v in self.values) + ">"


def make_blockwise_spec(
    back_key_function: KeyFunction,
    function: Callable[..., Any],
    num_input_blocks: tuple[int, ...] = (1,),
    output_names: set[str] | None = None,
) -> BlockwiseSpec:
    output_names = output_names or {"out"}
    return BlockwiseSpec(
        back_key_function=back_key_function,
        function=function,
        num_input_blocks=num_input_blocks,
        num_output_blocks=(1,),  # not needed for this test
        reads_map={},  # not needed for this test
        writes_map={name: None for name in output_names},  # only used for names
    )


def test_map_blocks_back_key_function() -> None:
    back_key_function = make_map_blocks_back_key_function("a")

    check_back_key_function(back_key_function, (0,), "≪('a', 0)≫")
    check_back_key_function(back_key_function, (1,), "≪('a', 1)≫")


def test_map_blocks_multiple_inputs_back_key_function() -> None:
    back_key_function = make_map_blocks_back_key_function("a", "b")

    check_back_key_function(back_key_function, (0,), "≪('a', 0), ('b', 0)≫")
    check_back_key_function(back_key_function, (1,), "≪('a', 1), ('b', 1)≫")


def test_combine_blocks_list_back_key_function() -> None:
    back_key_function = make_combine_blocks_list_back_key_function(
        "a", numblocks=5, split_every=2
    )

    check_back_key_function(back_key_function, (0,), "≪[('a', 0), ('a', 1)]≫")
    check_back_key_function(back_key_function, (1,), "≪[('a', 2), ('a', 3)]≫")
    check_back_key_function(back_key_function, (2,), "≪[('a', 4)]≫")


def test_combine_blocks_iter_back_key_function() -> None:
    back_key_function = make_combine_blocks_iter_back_key_function(
        "a", numblocks=5, split_every=2
    )

    check_back_key_function(back_key_function, (0,), "≪<('a', 0), ('a', 1)>≫")
    check_back_key_function(back_key_function, (1,), "≪<('a', 2), ('a', 3)>≫")
    check_back_key_function(back_key_function, (2,), "≪<('a', 4)>≫")


def test_alternate_blocks_back_key_function() -> None:
    back_key_function = make_alternate_blocks_back_key_function("a", "b")

    check_back_key_function(back_key_function, (0,), "≪('a', 0)≫")
    check_back_key_function(back_key_function, (1,), "≪('b', 1)≫")
    check_back_key_function(back_key_function, (2,), "≪('a', 2)≫")
    check_back_key_function(back_key_function, (3,), "≪('b', 3)≫")


def test_concat_blocks_back_key_function() -> None:
    back_key_function = make_concat_blocks_back_key_function("a", "b")

    check_back_key_function(back_key_function, (0,), "≪<('a', 0)>≫")
    check_back_key_function(back_key_function, (1,), "≪<('a', 1), ('b', 0)>≫")
    check_back_key_function(back_key_function, (2,), "≪<('b', 0), ('b', 1)>≫")


def test_fuse_back_key_function_map_blocks_linear() -> None:
    #
    #   a
    #   |
    #   b
    #   |
    #   c
    #   |
    #  out
    #
    back_key_function1 = make_map_blocks_back_key_function("a")
    back_key_function2 = make_map_blocks_back_key_function("b")
    back_key_function3 = make_map_blocks_back_key_function("c")
    fused_back_key_function = make_fused_back_key_function(
        back_key_function2, {"b": back_key_function1}
    )
    fused_back_key_function = make_fused_back_key_function(
        back_key_function3, {"c": fused_back_key_function}
    )

    check_back_key_function(fused_back_key_function, (0,), "≪≪≪('a', 0)≫≫≫")
    check_back_key_function(fused_back_key_function, (1,), "≪≪≪('a', 1)≫≫≫")


def test_fuse_back_key_function_map_blocks_branching() -> None:
    #
    #  a   b c
    #   \ /  |
    #    d   e
    #     \ /
    #     out
    #
    back_key_function1 = make_map_blocks_back_key_function("a", "b")
    back_key_function2 = make_map_blocks_back_key_function("c")
    back_key_function3 = make_map_blocks_back_key_function("d", "e")
    fused_back_key_function = make_fused_back_key_function(
        back_key_function3, {"d": back_key_function1, "e": back_key_function2}
    )

    check_back_key_function(
        fused_back_key_function, (0,), "≪≪('a', 0), ('b', 0)≫, ≪('c', 0)≫≫"
    )
    check_back_key_function(
        fused_back_key_function, (1,), "≪≪('a', 1), ('b', 1)≫, ≪('c', 1)≫≫"
    )


def test_fuse_back_key_function_map_blocks_branching_mixed_levels() -> None:
    #
    #  a   b
    #   \ /
    #    c   d
    #     \ /
    #     out
    #
    back_key_function1 = make_map_blocks_back_key_function("a", "b")
    back_key_function2 = make_map_blocks_back_key_function("c", "d")
    fused_back_key_function = make_fused_back_key_function(
        back_key_function2, {"c": back_key_function1}
    )

    # note that d is wrapped in a function (the identity)
    check_back_key_function(
        fused_back_key_function, (0,), "≪≪('a', 0), ('b', 0)≫, ≪('d', 0)≫≫"
    )
    check_back_key_function(
        fused_back_key_function, (1,), "≪≪('a', 1), ('b', 1)≫, ≪('d', 1)≫≫"
    )


def test_fuse_back_key_function_map_blocks_combine_blocks_list() -> None:
    back_key_function1 = make_map_blocks_back_key_function("a")
    back_key_function2 = make_combine_blocks_list_back_key_function(
        "b", numblocks=5, split_every=2
    )
    fused_back_key_function = make_fused_back_key_function(
        back_key_function2, {"b": back_key_function1}
    )

    check_back_key_function(fused_back_key_function, (0,), "≪[≪('a', 0)≫, ≪('a', 1)≫]≫")
    check_back_key_function(fused_back_key_function, (1,), "≪[≪('a', 2)≫, ≪('a', 3)≫]≫")
    check_back_key_function(fused_back_key_function, (2,), "≪[≪('a', 4)≫]≫")


def test_fuse_back_key_function_map_blocks_combine_blocks_iter() -> None:
    back_key_function1 = make_map_blocks_back_key_function("a")
    back_key_function2 = make_combine_blocks_iter_back_key_function(
        "b", numblocks=5, split_every=2
    )
    fused_back_key_function = make_fused_back_key_function(
        back_key_function2, {"b": back_key_function1}
    )

    check_back_key_function(fused_back_key_function, (0,), "≪<≪('a', 0)≫, ≪('a', 1)≫>≫")
    check_back_key_function(fused_back_key_function, (1,), "≪<≪('a', 2)≫, ≪('a', 3)≫>≫")
    check_back_key_function(fused_back_key_function, (2,), "≪<≪('a', 4)≫>≫")


def test_fuse_back_key_function_combine_blocks_list_map_blocks() -> None:
    back_key_function1 = make_combine_blocks_list_back_key_function(
        "a", numblocks=5, split_every=2
    )
    back_key_function2 = make_map_blocks_back_key_function("b")
    fused_back_key_function = make_fused_back_key_function(
        back_key_function2, {"b": back_key_function1}
    )

    check_back_key_function(fused_back_key_function, (0,), "≪≪[('a', 0), ('a', 1)]≫≫")
    check_back_key_function(fused_back_key_function, (1,), "≪≪[('a', 2), ('a', 3)]≫≫")
    check_back_key_function(fused_back_key_function, (2,), "≪≪[('a', 4)]≫≫")


def test_fuse_back_key_function_combine_blocks_iter_map_blocks() -> None:
    back_key_function1 = make_combine_blocks_iter_back_key_function(
        "a", numblocks=5, split_every=2
    )
    back_key_function2 = make_map_blocks_back_key_function("b")
    fused_back_key_function = make_fused_back_key_function(
        back_key_function2, {"b": back_key_function1}
    )

    check_back_key_function(fused_back_key_function, (0,), "≪≪<('a', 0), ('a', 1)>≫≫")
    check_back_key_function(fused_back_key_function, (1,), "≪≪<('a', 2), ('a', 3)>≫≫")
    check_back_key_function(fused_back_key_function, (2,), "≪≪<('a', 4)>≫≫")


def test_fuse_back_key_function_combine_blocks_list_combine_blocks_list() -> None:
    back_key_function1 = make_combine_blocks_list_back_key_function(
        "a", numblocks=5, split_every=2
    )
    back_key_function2 = make_combine_blocks_list_back_key_function(
        "b", numblocks=3, split_every=2
    )
    fused_back_key_function = make_fused_back_key_function(
        back_key_function2, {"b": back_key_function1}
    )

    check_back_key_function(
        fused_back_key_function,
        (0,),
        "≪[≪[('a', 0), ('a', 1)]≫, ≪[('a', 2), ('a', 3)]≫]≫",
    )
    check_back_key_function(fused_back_key_function, (1,), "≪[≪[('a', 4)]≫]≫")


def test_fuse_back_key_function_combine_blocks_iter_combine_blocks_iter() -> None:
    back_key_function1 = make_combine_blocks_iter_back_key_function(
        "a", numblocks=5, split_every=2
    )
    back_key_function2 = make_combine_blocks_iter_back_key_function(
        "b", numblocks=3, split_every=2
    )
    fused_back_key_function = make_fused_back_key_function(
        back_key_function2, {"b": back_key_function1}
    )

    check_back_key_function(
        fused_back_key_function,
        (0,),
        "≪<≪<('a', 0), ('a', 1)>≫, ≪<('a', 2), ('a', 3)>≫>≫",
    )
    check_back_key_function(fused_back_key_function, (1,), "≪<≪<('a', 4)>≫>≫")


def test_fuse_back_key_function_map_blocks_alternate_blocks_back_key_function() -> None:
    #
    #  a   b
    #  |   |
    #  c   d
    #   \ /
    #   out
    #
    back_key_function1 = make_map_blocks_back_key_function("a")
    back_key_function2 = make_map_blocks_back_key_function("b")
    back_key_function3 = make_alternate_blocks_back_key_function("c", "d")
    fused_back_key_function = make_fused_back_key_function(
        back_key_function3, {"c": back_key_function1, "d": back_key_function2}
    )

    check_back_key_function(fused_back_key_function, (0,), "≪≪('a', 0)≫≫")
    check_back_key_function(fused_back_key_function, (1,), "≪≪('b', 1)≫≫")
    check_back_key_function(fused_back_key_function, (2,), "≪≪('a', 2)≫≫")
    check_back_key_function(fused_back_key_function, (3,), "≪≪('b', 3)≫≫")


def test_fuse_back_key_function_map_blocks_concat_blocks_back_key_function() -> None:
    #
    #  a   b
    #  |   |
    #  c   d
    #   \ /
    #   out
    #
    back_key_function1 = make_map_blocks_back_key_function("a")
    back_key_function2 = make_map_blocks_back_key_function("b")
    back_key_function3 = make_concat_blocks_back_key_function("c", "d")
    fused_back_key_function = make_fused_back_key_function(
        back_key_function3, {"c": back_key_function1, "d": back_key_function2}
    )

    check_back_key_function(fused_back_key_function, (0,), "≪<≪('a', 0)≫>≫")
    check_back_key_function(fused_back_key_function, (1,), "≪<≪('a', 1)≫, ≪('b', 0)≫>≫")
    check_back_key_function(fused_back_key_function, (2,), "≪<≪('b', 0)≫, ≪('b', 1)≫>≫")


def apply_blockwise(
    input_data: dict[str, list[int]], out_coords: list[int], bw_spec: BlockwiseSpec
) -> Any:
    # just return the (1D) coord as a value
    def get_data(key: ChunkKey) -> int:
        logger.debug(f"get_data for f{key}")
        name = key.name
        index = key.coords[0]  # 1d index
        return input_data[name][index]

    logger.debug("Calling apply_blockwise...")
    out_key = ChunkKey(
        "out", tuple(out_coords)
    )  # array name is ignored by back_key_function
    logger.debug("out_key: %s", out_key)
    in_keys = bw_spec.back_key_function(out_key)
    logger.debug("in_keys: %s", in_keys)
    fargs = map_nested(get_data, in_keys)
    logger.debug("fargs: %s", fargs)
    return bw_spec.function(*fargs.args)


def test_apply_blockwise() -> None:
    bw_spec = make_blockwise_spec(
        back_key_function=make_map_blocks_back_key_function("a"),
        function=negative,
    )

    input_data = {"a": [0, 1, 2, 3, 4]}
    out = [apply_blockwise(input_data, [i], bw_spec) for i in range(5)]
    assert out == [0, -1, -2, -3, -4]


def test_apply_blockwise_multiple_inputs() -> None:
    bw_spec = make_blockwise_spec(
        back_key_function=make_map_blocks_back_key_function("a", "b"), function=add
    )

    input_data = {"a": [0, 1, 2, 3, 4], "b": [5, 6, 7, 8, 9]}
    out = [apply_blockwise(input_data, [i], bw_spec) for i in range(5)]
    assert out == [5, 7, 9, 11, 13]


def test_apply_blockwise_iterator() -> None:
    bw_spec = make_blockwise_spec(
        back_key_function=make_combine_blocks_iter_back_key_function(
            "a", numblocks=5, split_every=2
        ),
        function=sum_iter,
    )

    input_data = {"a": [0, 1, 2, 3, 4]}
    out = [apply_blockwise(input_data, [i], bw_spec) for i in range(3)]
    assert out == [1, 5, 4]


def test_apply_blockwise_multiple_outputs() -> None:
    bw_spec = make_blockwise_spec(
        back_key_function=make_map_blocks_back_key_function("a"),
        function=int_sqrts,
    )

    input_data = {"a": [0, 1, 4, 9, 16]}
    out = [apply_blockwise(input_data, [i], bw_spec) for i in range(5)]
    vals = [tuple(y for y in x) for x in out]
    assert vals == [(0, 0), (1, -1), (2, -2), (3, -3), (4, -4)]


def test_apply_blockwise_fused() -> None:
    bw_spec1 = make_blockwise_spec(
        back_key_function=make_map_blocks_back_key_function("a"),
        function=negative,
        output_names={"b"},
    )
    bw_spec2 = make_blockwise_spec(
        back_key_function=make_map_blocks_back_key_function("b"), function=negative
    )

    bw_spec = fuse_blockwise_specs(bw_spec2, bw_spec1)

    input_data = {"a": [0, 1, 2, 3, 4]}
    out = [apply_blockwise(input_data, [i], bw_spec) for i in range(5)]
    assert out == [0, 1, 2, 3, 4]


def test_apply_blockwise_fused_list() -> None:
    bw_spec1 = make_blockwise_spec(
        back_key_function=make_map_blocks_back_key_function("a"),
        function=negative,
        output_names={"b"},
    )
    bw_spec2 = make_blockwise_spec(
        back_key_function=make_combine_blocks_list_back_key_function(
            "b", numblocks=5, split_every=2
        ),
        function=sum_list,
    )

    bw_spec = fuse_blockwise_specs(bw_spec2, bw_spec1)

    input_data = {"a": [0, 1, 2, 3, 4]}
    out = [apply_blockwise(input_data, [i], bw_spec) for i in range(3)]
    assert out == [-1, -5, -4]


def test_apply_blockwise_fused_iterator() -> None:
    bw_spec1 = make_blockwise_spec(
        back_key_function=make_map_blocks_back_key_function("a"),
        function=negative,
        output_names={"b"},
    )
    bw_spec2 = make_blockwise_spec(
        back_key_function=make_combine_blocks_iter_back_key_function(
            "b", numblocks=5, split_every=2
        ),
        function=sum_iter,
    )

    bw_spec = fuse_blockwise_specs(bw_spec2, bw_spec1)

    input_data = {"a": [0, 1, 2, 3, 4]}
    out = [apply_blockwise(input_data, [i], bw_spec) for i in range(3)]
    assert out == [-1, -5, -4]


def test_apply_blockwise_fused_iterator_with_single_input_block() -> None:
    bw_spec1 = make_blockwise_spec(
        back_key_function=make_map_blocks_back_key_function("a"),
        function=negative,
        output_names={"b"},
    )
    # the iterator only reads from a single input block
    bw_spec2 = make_blockwise_spec(
        back_key_function=make_combine_blocks_iter_back_key_function(
            "b", numblocks=5, split_every=1
        ),
        function=sum_iter,
    )

    bw_spec = fuse_blockwise_specs(bw_spec2, bw_spec1)

    input_data = {"a": [0, 1, 2, 3, 4]}
    out = [apply_blockwise(input_data, [i], bw_spec) for i in range(5)]
    assert out == [0, -1, -2, -3, -4]


def test_apply_blockwise_fused_iterators() -> None:
    bw_spec1 = make_blockwise_spec(
        back_key_function=make_combine_blocks_iter_back_key_function(
            "a", numblocks=5, split_every=2
        ),
        function=sum_iter,
        output_names={"b"},
    )
    bw_spec2 = make_blockwise_spec(
        back_key_function=make_combine_blocks_iter_back_key_function(
            "b", numblocks=3, split_every=2
        ),
        function=sum_iter,
    )

    bw_spec = fuse_blockwise_specs(bw_spec2, bw_spec1)

    input_data = {"a": [0, 1, 2, 3, 4]}
    out = [apply_blockwise(input_data, [i], bw_spec) for i in range(2)]
    assert out == [6, 4]


def test_apply_blockwise_alternate_blocks_fused() -> None:
    bw_spec1 = make_blockwise_spec(
        back_key_function=make_map_blocks_back_key_function("a"),
        function=negative,
        output_names={"c"},
    )
    bw_spec2 = make_blockwise_spec(
        back_key_function=make_map_blocks_back_key_function("b"),
        function=identity,
        output_names={"d"},
    )
    bw_spec3 = make_blockwise_spec(
        back_key_function=make_alternate_blocks_back_key_function("c", "d"),
        function=identity,
        num_input_blocks=(1, 1),
    )

    bw_spec = fuse_blockwise_specs(bw_spec3, bw_spec1, bw_spec2)

    input_data = {"a": [0, 2, 4, 6], "b": [1, 3, 5, 7]}
    out = [apply_blockwise(input_data, [i], bw_spec) for i in range(4)]
    assert out == [0, 3, -4, 7]


def test_apply_blockwise_concat_blocks_fused() -> None:
    bw_spec1 = make_blockwise_spec(
        back_key_function=make_map_blocks_back_key_function("a"),
        function=negative,
        output_names={"c"},
    )
    bw_spec2 = make_blockwise_spec(
        back_key_function=make_map_blocks_back_key_function("b"),
        function=identity,
        output_names={"d"},
    )
    bw_spec3 = make_blockwise_spec(
        back_key_function=make_concat_blocks_back_key_function("c", "d"),
        function=sum_iter,
        num_input_blocks=(1, 1),
    )

    bw_spec = fuse_blockwise_specs(bw_spec3, bw_spec1, bw_spec2)

    input_data = {"a": [0, 1], "b": [2, 3]}
    out = [apply_blockwise(input_data, [i], bw_spec) for i in range(3)]
    assert out == [0, -1 + 2, 2 + 3]


def test_apply_blockwise_multiple_outputs_fused() -> None:
    bw_spec1 = make_blockwise_spec(
        back_key_function=make_map_blocks_back_key_function("a"),
        function=negative,
        output_names={"b"},
    )
    bw_spec2 = make_blockwise_spec(
        back_key_function=make_map_blocks_back_key_function("b"),
        function=int_sqrts,
    )

    bw_spec = fuse_blockwise_specs(bw_spec2, bw_spec1)

    input_data = {"a": [0, -1, -4, -9, -16]}
    out = [apply_blockwise(input_data, [i], bw_spec) for i in range(5)]
    vals = [tuple(y for y in x) for x in out]
    assert vals == [(0, 0), (1, -1), (2, -2), (3, -3), (4, -4)]
