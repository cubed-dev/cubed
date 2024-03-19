from functools import partial

import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_array_equal

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.backend_array_api import namespace as nxp
from cubed.core.ops import elemwise, merge_chunks_new, partial_reduce
from cubed.core.optimization import (
    fuse_all_optimize_dag,
    fuse_only_optimize_dag,
    fuse_predecessors,
    gensym,
    multiple_inputs_optimize_dag,
    simple_optimize_dag,
)
from cubed.core.plan import arrays_to_plan
from cubed.tests.utils import TaskCounter


@pytest.fixture()
def spec(tmp_path):
    return cubed.Spec(tmp_path, allowed_mem=100000)


def test_fusion(spec):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.negative(a)
    c = xp.astype(b, np.float32)
    d = xp.negative(c)

    num_arrays = 4  # a, b, c, d
    num_created_arrays = 3  # b, c, d (a is not created on disk)
    assert d.plan.num_arrays(optimize_graph=False) == num_arrays
    assert d.plan.num_tasks(optimize_graph=False) == num_created_arrays + 12
    assert (
        d.plan.total_nbytes_written(optimize_graph=False)
        == b.nbytes + c.nbytes + d.nbytes
    )
    num_arrays = 2  # a, d
    num_created_arrays = 1  # d (a is not created on disk)
    assert d.plan.num_arrays(optimize_graph=True) == num_arrays
    assert d.plan.num_tasks(optimize_graph=True) == num_created_arrays + 4
    assert d.plan.total_nbytes_written(optimize_graph=True) == d.nbytes

    task_counter = TaskCounter()
    result = d.compute(callbacks=[task_counter])
    assert task_counter.value == num_created_arrays + 4

    assert_array_equal(
        result,
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32),
    )


def test_fusion_transpose(spec):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.negative(a)
    c = xp.astype(b, np.float32)
    d = c.T

    num_created_arrays = 3  # b, c, d
    assert d.plan.num_tasks(optimize_graph=False) == num_created_arrays + 12
    num_created_arrays = 1  # d
    assert d.plan.num_tasks(optimize_graph=True) == num_created_arrays + 4

    task_counter = TaskCounter()
    result = d.compute(callbacks=[task_counter])
    assert task_counter.value == num_created_arrays + 4

    assert_array_equal(
        result,
        np.array([[-1, -4, -7], [-2, -5, -8], [-3, -6, -9]]).astype(np.float32),
    )


def test_no_fusion(spec):
    # b can't be fused with c because d also depends on b
    a = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    b = xp.positive(a)
    c = xp.positive(b)
    d = xp.equal(b, c)

    opt_fn = simple_optimize_dag

    num_created_arrays = 3  # b, c, d
    assert d.plan.num_tasks(optimize_graph=False) == num_created_arrays + 3
    assert d.plan.num_tasks(optimize_function=opt_fn) == num_created_arrays + 3

    task_counter = TaskCounter()
    result = d.compute(optimize_function=opt_fn, callbacks=[task_counter])
    assert task_counter.value == num_created_arrays + 3

    assert_array_equal(result, np.ones((2, 2)))


def test_no_fusion_multiple_edges(spec):
    a = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    b = xp.positive(a)
    c = xp.asarray(b)
    # b and c are the same array, so d has a single dependency
    # with multiple edges
    # this should not be fused under the current logic
    d = xp.equal(b, c)

    opt_fn = simple_optimize_dag

    num_created_arrays = 2  # c, d
    assert d.plan.num_tasks(optimize_graph=False) == num_created_arrays + 2
    assert d.plan.num_tasks(optimize_function=opt_fn) == num_created_arrays + 2

    task_counter = TaskCounter()
    result = d.compute(optimize_function=opt_fn, callbacks=[task_counter])
    assert task_counter.value == num_created_arrays + 2

    assert_array_equal(result, np.full((2, 2), True))


def test_custom_optimize_function(spec):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.negative(a)
    c = xp.astype(b, np.float32)
    d = xp.negative(c)

    num_tasks_with_no_optimization = d.plan.num_tasks(optimize_graph=False)

    assert d.plan.num_tasks(optimize_graph=True) < num_tasks_with_no_optimization

    def custom_optimize_function(dag):
        # leave DAG unchanged
        return dag

    assert (
        d.plan.num_tasks(
            optimize_graph=True, optimize_function=custom_optimize_function
        )
        == num_tasks_with_no_optimization
    )


def get_num_input_blocks(dag, arr_name):
    op_name = next(dag.predecessors(arr_name))
    return dag.nodes(data=True)[op_name]["pipeline"].config.num_input_blocks


def fuse_one_level(arr, *, always_fuse=None):
    # use fuse_predecessors to test one level of fusion
    return partial(
        fuse_predecessors,
        name=next(arr.plan.dag.predecessors(arr.name)),
        always_fuse=always_fuse,
    )


def fuse_multiple_levels(*, max_total_source_arrays=4, max_total_num_input_blocks=None):
    # use multiple_inputs_optimize_dag to test multiple levels of fusion
    return partial(
        multiple_inputs_optimize_dag,
        max_total_source_arrays=max_total_source_arrays,
        max_total_num_input_blocks=max_total_num_input_blocks,
    )


# utility functions for testing structural equivalence of dags


def create_dag():
    return nx.MultiDiGraph()


def add_op(dag, func, inputs, outputs, fusable=True):
    name = gensym(func.__name__)
    dag.add_node(name, func=func, fusable=fusable)
    for n in inputs:
        dag.add_edge(n, name)
    for n in outputs:
        dag.add_node(n)
        dag.add_edge(name, n)

    return name


def placeholder_func(*x):
    return 1


def add_placeholder_op(dag, inputs, outputs):
    add_op(dag, placeholder_func, [a.name for a in inputs], [b.name for b in outputs])


def structurally_equivalent(dag1, dag2):
    # compare structure, and node labels for values but not operators since they are placeholders

    # draw_dag(dag1, "dag1")  # uncomment for debugging
    # draw_dag(dag2, "dag2")  # uncomment for debugging

    labelled_dag1 = nx.convert_node_labels_to_integers(dag1, label_attribute="label")
    labelled_dag2 = nx.convert_node_labels_to_integers(dag2, label_attribute="label")

    def nm(node_attrs1, node_attrs2):
        label1 = node_attrs1["label"]
        label2 = node_attrs2["label"]
        # - in a label indicates that the node is an operator; don't compare these names
        if "-" in label1:
            return "-" in label2
        return label1 == label2

    return nx.is_isomorphic(labelled_dag1, labelled_dag2, node_match=nm)


def draw_dag(dag, name="dag"):
    dag = dag.copy()
    for _, d in dag.nodes(data=True):
        if "name" in d:  # pydot already has name
            del d["name"]
    gv = nx.drawing.nx_pydot.to_pydot(dag)
    format = "svg"
    full_filename = f"{name}.{format}"
    gv.write(full_filename, format=format)


# simple unary function
#
#  a        ->       a
#  |                 |
#  b                 c
#  |
#  c
#
def test_fuse_unary_op(spec):
    a = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    b = xp.negative(a)
    c = xp.negative(b)

    opt_fn = fuse_one_level(c)

    c.visualize(optimize_function=opt_fn)

    # check structure of optimized dag
    expected_fused_dag = create_dag()
    add_placeholder_op(expected_fused_dag, (), (a,))
    add_placeholder_op(expected_fused_dag, (a,), (c,))
    optimized_dag = c.plan.optimize(optimize_function=opt_fn).dag
    assert structurally_equivalent(optimized_dag, expected_fused_dag)
    assert get_num_input_blocks(c.plan.dag, c.name) == (1,)
    assert get_num_input_blocks(optimized_dag, c.name) == (1,)

    num_created_arrays = 2  # b, c
    assert c.plan.num_tasks(optimize_graph=False) == num_created_arrays + 2
    num_created_arrays = 1  # c
    assert c.plan.num_tasks(optimize_function=opt_fn) == num_created_arrays + 1

    task_counter = TaskCounter()
    result = c.compute(callbacks=[task_counter], optimize_function=opt_fn)
    assert task_counter.value == num_created_arrays + 1

    assert_array_equal(result, np.ones((2, 2)))


# simple binary function
#
#  a   b    ->   a   b
#  |   |          \ /
#  c   d           e
#   \ /
#    e
#
def test_fuse_binary_op(spec):
    a = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    b = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    c = xp.negative(a)
    d = xp.negative(b)
    e = xp.add(c, d)

    opt_fn = fuse_one_level(e)

    e.visualize(optimize_function=opt_fn)

    # check structure of optimized dag
    expected_fused_dag = create_dag()
    add_placeholder_op(expected_fused_dag, (), (a,))
    add_placeholder_op(expected_fused_dag, (), (b,))
    add_placeholder_op(expected_fused_dag, (a, b), (e,))
    optimized_dag = e.plan.optimize(optimize_function=opt_fn).dag
    assert structurally_equivalent(optimized_dag, expected_fused_dag)
    assert get_num_input_blocks(e.plan.dag, e.name) == (1, 1)
    assert get_num_input_blocks(optimized_dag, e.name) == (1, 1)

    num_created_arrays = 3  # c, d, e
    assert e.plan.num_tasks(optimize_graph=False) == num_created_arrays + 3
    num_created_arrays = 1  # e
    assert e.plan.num_tasks(optimize_function=opt_fn) == num_created_arrays + 1

    task_counter = TaskCounter()
    result = e.compute(callbacks=[task_counter], optimize_function=opt_fn)
    assert task_counter.value == num_created_arrays + 1

    assert_array_equal(result, -2 * np.ones((2, 2)))


# unary and binary functions
#
#  a b   c    ->   a  b  c
#  |  \ /           \ | /
#  d   e              f
#   \ /
#    f
#
def test_fuse_unary_and_binary_op(spec):
    a = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    b = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    c = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    d = xp.negative(a)
    e = xp.add(b, c)
    f = xp.add(d, e)

    opt_fn = fuse_one_level(f)

    f.visualize(optimize_function=opt_fn)

    # check structure of optimized dag
    expected_fused_dag = create_dag()
    add_placeholder_op(expected_fused_dag, (), (a,))
    add_placeholder_op(expected_fused_dag, (), (b,))
    add_placeholder_op(expected_fused_dag, (), (c,))
    add_placeholder_op(expected_fused_dag, (a, b, c), (f,))
    optimized_dag = f.plan.optimize(optimize_function=opt_fn).dag
    assert structurally_equivalent(optimized_dag, expected_fused_dag)
    assert get_num_input_blocks(f.plan.dag, f.name) == (1, 1)
    assert get_num_input_blocks(optimized_dag, f.name) == (1, 1, 1)

    result = f.compute(optimize_function=opt_fn)
    assert_array_equal(result, np.ones((2, 2)))


# mixed levels
#
#    b   c    ->   a  b  c
#     \ /           \ | /
#  a   d              e
#   \ /
#    e
#
def test_fuse_mixed_levels(spec):
    a = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    b = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    c = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    d = xp.add(b, c)
    e = xp.add(a, d)

    opt_fn = fuse_one_level(e)

    e.visualize(optimize_function=opt_fn)

    # check structure of optimized dag
    expected_fused_dag = create_dag()
    add_placeholder_op(expected_fused_dag, (), (a,))
    add_placeholder_op(expected_fused_dag, (), (b,))
    add_placeholder_op(expected_fused_dag, (), (c,))
    add_placeholder_op(expected_fused_dag, (a, b, c), (e,))
    optimized_dag = e.plan.optimize(optimize_function=opt_fn).dag
    assert structurally_equivalent(optimized_dag, expected_fused_dag)
    assert get_num_input_blocks(e.plan.dag, e.name) == (1, 1)
    assert get_num_input_blocks(optimized_dag, e.name) == (1, 1, 1)

    result = e.compute(optimize_function=opt_fn)
    assert_array_equal(result, 3 * np.ones((2, 2)))


# diamond
#
#   a    ->    a
#  / \         ‖
# b   c        d
#  \ /
#   d
#
def test_fuse_diamond(spec):
    a = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    b = xp.positive(a)
    c = xp.positive(a)
    d = xp.add(b, c)

    opt_fn = fuse_one_level(d)

    d.visualize(optimize_function=opt_fn)

    # check structure of optimized dag
    expected_fused_dag = create_dag()
    add_placeholder_op(expected_fused_dag, (), (a,))
    add_placeholder_op(expected_fused_dag, (a, a), (d,))
    optimized_dag = d.plan.optimize(optimize_function=opt_fn).dag
    assert structurally_equivalent(optimized_dag, expected_fused_dag)
    assert get_num_input_blocks(d.plan.dag, d.name) == (1, 1)
    assert get_num_input_blocks(optimized_dag, d.name) == (1, 1)

    result = d.compute(optimize_function=opt_fn)
    assert_array_equal(result, 2 * np.ones((2, 2)))


# mixed levels and diamond
# from https://github.com/cubed-dev/cubed/issues/126
#
#   a    ->    a
#   |         /|
#   b        b |
#  /|         \|
# c |          d
#  \|
#   d
#
def test_fuse_mixed_levels_and_diamond(spec):
    a = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    b = xp.positive(a)
    c = xp.positive(b)
    d = xp.add(b, c)

    opt_fn = fuse_one_level(d)

    d.visualize(optimize_function=opt_fn)

    # check structure of optimized dag
    expected_fused_dag = create_dag()
    add_placeholder_op(expected_fused_dag, (), (a,))
    add_placeholder_op(expected_fused_dag, (a,), (b,))
    add_placeholder_op(expected_fused_dag, (a, b), (d,))
    optimized_dag = d.plan.optimize(optimize_function=opt_fn).dag
    assert structurally_equivalent(optimized_dag, expected_fused_dag)
    assert get_num_input_blocks(d.plan.dag, d.name) == (1, 1)
    assert get_num_input_blocks(optimized_dag, d.name) == (1, 1)

    result = d.compute(optimize_function=opt_fn)
    assert_array_equal(result, 2 * np.ones((2, 2)))


# derived from a bug found by array_api_tests/test_manipulation_functions.py::test_expand_dims
#  a   b    ->   a   b
#   \ /          |\ /|
#    c           | d |
#   /|           | | |
#  d |           | e |
#  | |            \|/
#  e |             f
#   \|
#    f
def test_fuse_mixed_levels_and_diamond_complex(spec):
    a = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    b = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    c = xp.add(a, b)
    d = xp.positive(c)
    e = d[1:, :]  # operation can't be fused
    f = xp.add(e, c)  # this order exposed a bug in argument ordering

    opt_fn = multiple_inputs_optimize_dag

    f.visualize(optimize_function=opt_fn)
    result = f.compute(optimize_function=opt_fn)
    assert_array_equal(result, 4 * np.ones((2, 2)))


# repeated argument
# from https://github.com/cubed-dev/cubed/issues/65
#
#  a   ->   a
#  |        ‖
#  b        c
#  ‖
#  c
#
def test_fuse_repeated_argument(spec):
    a = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    b = xp.negative(a)
    c = xp.add(b, b)

    opt_fn = fuse_one_level(c)

    c.visualize(optimize_function=opt_fn)

    # check structure of optimized dag
    expected_fused_dag = create_dag()
    add_placeholder_op(expected_fused_dag, (), (a,))
    add_placeholder_op(expected_fused_dag, (a, a), (c,))
    optimized_dag = c.plan.optimize(optimize_function=opt_fn).dag
    assert structurally_equivalent(optimized_dag, expected_fused_dag)
    assert get_num_input_blocks(c.plan.dag, c.name) == (1, 1)
    assert get_num_input_blocks(optimized_dag, c.name) == (1, 1)

    result = c.compute(optimize_function=opt_fn)
    assert_array_equal(result, -2 * np.ones((2, 2)))


# other dependents
#
#   a    ->    a
#   |         / \
#   b        c   b
#  / \           |
# c   d          d
#
def test_fuse_other_dependents(spec):
    a = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    b = xp.negative(a)
    c = xp.negative(b)
    d = xp.negative(b)

    # only fuse c; leave d unfused
    opt_fn = fuse_one_level(c)

    # note multi-arg forms of visualize and compute below
    cubed.visualize(c, d, optimize_function=opt_fn)

    # check structure of optimized dag
    expected_fused_dag = create_dag()
    add_placeholder_op(expected_fused_dag, (), (a,))
    add_placeholder_op(expected_fused_dag, (a,), (b,))
    add_placeholder_op(expected_fused_dag, (a,), (c,))
    add_placeholder_op(expected_fused_dag, (b,), (d,))
    plan = arrays_to_plan(c, d)
    optimized_dag = plan.optimize(optimize_function=opt_fn).dag
    assert structurally_equivalent(optimized_dag, expected_fused_dag)
    assert get_num_input_blocks(c.plan.dag, c.name) == (1,)
    assert get_num_input_blocks(optimized_dag, c.name) == (1,)

    c_result, d_result = cubed.compute(c, d, optimize_function=opt_fn)
    assert_array_equal(c_result, np.ones((2, 2)))
    assert_array_equal(d_result, np.ones((2, 2)))


# unary large fan-in
#
#  a b c d         e f g h   ->   a b c d         e f g h
#   \ \ \ \       / / / /          \ \ \ \       / / / /
#    \ \ \ \     / / / /            \ \ \ \     / / / /
#     \ \ \ \   / / / /              \ \ \ \   / / / /
#      \ \ \ \ / / / /                \ \ \ \ / / / /
#       ----- i -----                  ----- j -----
#             |
#             j
#
def test_fuse_unary_large_fan_in(spec):
    a = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    b = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    c = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    d = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    e = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    f = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    g = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    h = xp.ones((2, 2), chunks=(2, 2), spec=spec)

    # use elemwise and stack since add can only take 2 args
    def stack_add(*a):
        return nxp.sum(nxp.stack(a), axis=0)

    i = elemwise(stack_add, a, b, c, d, e, f, g, h, dtype=a.dtype)
    j = xp.negative(i)

    # max_total_source_arrays is left at its default (4) which does not limit fusion since j is unary
    opt_fn = fuse_one_level(j)

    j.visualize(optimize_function=opt_fn)

    # check structure of optimized dag
    expected_fused_dag = create_dag()
    add_placeholder_op(expected_fused_dag, (), (a,))
    add_placeholder_op(expected_fused_dag, (), (b,))
    add_placeholder_op(expected_fused_dag, (), (c,))
    add_placeholder_op(expected_fused_dag, (), (d,))
    add_placeholder_op(expected_fused_dag, (), (e,))
    add_placeholder_op(expected_fused_dag, (), (f,))
    add_placeholder_op(expected_fused_dag, (), (g,))
    add_placeholder_op(expected_fused_dag, (), (h,))
    add_placeholder_op(
        expected_fused_dag,
        (
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            h,
        ),
        (j,),
    )
    optimized_dag = j.plan.optimize(optimize_function=opt_fn).dag
    assert structurally_equivalent(optimized_dag, expected_fused_dag)
    assert get_num_input_blocks(j.plan.dag, j.name) == (1,)
    assert get_num_input_blocks(optimized_dag, j.name) == (1,) * 8

    result = j.compute(optimize_function=opt_fn)
    assert_array_equal(result, -8 * np.ones((2, 2)))


# large fan-in default
#
#  a   b c   d e   f g   h   ->   a  b   c  d e  f   g  h
#   \ /   \ /   \ /   \ /          \  \ /  /   \  \ /  /
#    i     j     k     m            -- n --     -- o --
#     \   /       \   /                 \         /
#       n           o                    \       /
#        \         /                      -- p --
#         \       /
#          \     /
#             p
#
def test_fuse_large_fan_in_default(spec):
    a = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    b = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    c = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    d = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    e = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    f = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    g = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    h = xp.ones((2, 2), chunks=(2, 2), spec=spec)

    i = xp.add(a, b)
    j = xp.add(c, d)
    k = xp.add(e, f)
    m = xp.add(g, h)

    n = xp.add(i, j)
    o = xp.add(k, m)

    p = xp.add(n, o)

    # max_total_source_arrays is left at its default (4) so only one level is fused
    opt_fn = fuse_multiple_levels()

    p.visualize(optimize_function=opt_fn)

    # check structure of optimized dag
    expected_fused_dag = create_dag()
    add_placeholder_op(expected_fused_dag, (), (a,))
    add_placeholder_op(expected_fused_dag, (), (b,))
    add_placeholder_op(expected_fused_dag, (), (c,))
    add_placeholder_op(expected_fused_dag, (), (d,))
    add_placeholder_op(expected_fused_dag, (), (e,))
    add_placeholder_op(expected_fused_dag, (), (f,))
    add_placeholder_op(expected_fused_dag, (), (g,))
    add_placeholder_op(expected_fused_dag, (), (h,))
    add_placeholder_op(expected_fused_dag, (a, b, c, d), (n,))
    add_placeholder_op(expected_fused_dag, (e, f, g, h), (o,))
    add_placeholder_op(expected_fused_dag, (n, o), (p,))
    optimized_dag = p.plan.optimize(optimize_function=opt_fn).dag
    assert structurally_equivalent(optimized_dag, expected_fused_dag)
    assert get_num_input_blocks(p.plan.dag, p.name) == (1, 1)
    assert get_num_input_blocks(optimized_dag, p.name) == (1, 1)

    result = p.compute(optimize_function=opt_fn)
    assert_array_equal(result, 8 * np.ones((2, 2)))


# large fan-in override
#
#  a   b c   d e   f g   h   ->   a b c d         e f g h
#   \ /   \ /   \ /   \ /          \ \ \ \       / / / /
#    i     j     k     m            \ \ \ \     / / / /
#     \   /       \   /              \ \ \ \   / / / /
#       n           o                 \ \ \ \ / / / /
#        \         /                   ----- p -----
#         \       /
#          \     /
#             p
#
def test_fuse_large_fan_in_override(spec):
    a = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    b = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    c = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    d = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    e = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    f = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    g = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    h = xp.ones((2, 2), chunks=(2, 2), spec=spec)

    i = xp.add(a, b)
    j = xp.add(c, d)
    k = xp.add(e, f)
    m = xp.add(g, h)

    n = xp.add(i, j)
    o = xp.add(k, m)

    p = xp.add(n, o)

    # max_total_source_arrays is overridden so multiple levels are fused
    opt_fn = fuse_multiple_levels(max_total_source_arrays=8)

    p.visualize(optimize_function=opt_fn)

    # check structure of optimized dag
    expected_fused_dag = create_dag()
    add_placeholder_op(expected_fused_dag, (), (a,))
    add_placeholder_op(expected_fused_dag, (), (b,))
    add_placeholder_op(expected_fused_dag, (), (c,))
    add_placeholder_op(expected_fused_dag, (), (d,))
    add_placeholder_op(expected_fused_dag, (), (e,))
    add_placeholder_op(expected_fused_dag, (), (f,))
    add_placeholder_op(expected_fused_dag, (), (g,))
    add_placeholder_op(expected_fused_dag, (), (h,))
    add_placeholder_op(
        expected_fused_dag,
        (
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            h,
        ),
        (p,),
    )
    optimized_dag = p.plan.optimize(optimize_function=opt_fn).dag
    assert structurally_equivalent(optimized_dag, expected_fused_dag)
    assert get_num_input_blocks(p.plan.dag, p.name) == (1, 1)
    assert get_num_input_blocks(optimized_dag, p.name) == (1,) * 8

    result = p.compute(optimize_function=opt_fn)
    assert_array_equal(result, 8 * np.ones((2, 2)))

    # now force everything to be fused with fuse_all_optimize_dag
    # note that max_total_source_arrays is *not* set
    opt_fn = fuse_all_optimize_dag
    optimized_dag = p.plan.optimize(optimize_function=opt_fn).dag
    assert structurally_equivalent(optimized_dag, expected_fused_dag)

    result = p.compute(optimize_function=opt_fn)
    assert_array_equal(result, 8 * np.ones((2, 2)))


# merge chunks with same number of tasks (unary)
#
#  a        ->       a
#  | 3               | 3
#  b                 c
#  | 1
#  c
#
def test_fuse_with_merge_chunks_unary(spec):
    a = xp.ones((3, 2), chunks=(1, 2), spec=spec)
    b = merge_chunks_new(a, chunks=(3, 2))
    c = xp.negative(b)

    opt_fn = fuse_one_level(c)

    c.visualize(optimize_function=opt_fn)

    # check structure of optimized dag
    expected_fused_dag = create_dag()
    add_placeholder_op(expected_fused_dag, (), (a,))
    add_placeholder_op(expected_fused_dag, (a,), (c,))
    optimized_dag = c.plan.optimize(optimize_function=opt_fn).dag
    assert structurally_equivalent(optimized_dag, expected_fused_dag)
    assert get_num_input_blocks(b.plan.dag, b.name) == (3,)
    assert get_num_input_blocks(c.plan.dag, c.name) == (1,)
    assert get_num_input_blocks(optimized_dag, c.name) == (3,)

    result = c.compute(optimize_function=opt_fn)
    assert_array_equal(result, -np.ones((3, 2)))


# merge chunks with same number of tasks (binary)
#
#   a   b    ->   a   b
# 3 |   | 1      3 \ / 1
#   c   d           e
#  1 \ / 1
#     e
#
def test_fuse_with_merge_chunks_binary(spec):
    a = xp.ones((3, 2), chunks=(1, 2), spec=spec)
    b = xp.ones((3, 2), chunks=(3, 2), spec=spec)
    c = merge_chunks_new(a, chunks=(3, 2))
    d = xp.negative(b)
    e = xp.add(c, d)

    opt_fn = fuse_one_level(e)

    e.visualize(optimize_function=opt_fn)

    # check structure of optimized dag
    expected_fused_dag = create_dag()
    add_placeholder_op(expected_fused_dag, (), (a,))
    add_placeholder_op(expected_fused_dag, (), (b,))
    add_placeholder_op(expected_fused_dag, (a, b), (e,))
    optimized_dag = e.plan.optimize(optimize_function=opt_fn).dag
    assert structurally_equivalent(optimized_dag, expected_fused_dag)
    assert get_num_input_blocks(e.plan.dag, e.name) == (1, 1)
    assert get_num_input_blocks(optimized_dag, e.name) == (3, 1)

    result = e.compute(optimize_function=opt_fn)
    assert_array_equal(result, np.zeros((3, 2)))


# merge chunks with different number of tasks (b has more tasks than c)
#
#  a        ->       a
#  | 1               | 3
#  b                 c
#  | 3
#  c
#
def test_fuse_merge_chunks_unary(spec):
    a = xp.ones((3, 2), chunks=(1, 2), spec=spec)
    b = xp.negative(a)
    c = merge_chunks_new(b, chunks=(3, 2))

    # specify max_total_num_input_blocks to force c to fuse
    opt_fn = fuse_multiple_levels(max_total_num_input_blocks=3)

    c.visualize(optimize_function=opt_fn)

    # check structure of optimized dag
    expected_fused_dag = create_dag()
    add_placeholder_op(expected_fused_dag, (), (a,))
    add_placeholder_op(expected_fused_dag, (a,), (c,))
    optimized_dag = c.plan.optimize(optimize_function=opt_fn).dag
    assert structurally_equivalent(optimized_dag, expected_fused_dag)
    assert get_num_input_blocks(b.plan.dag, b.name) == (1,)
    assert get_num_input_blocks(c.plan.dag, c.name) == (3,)
    assert get_num_input_blocks(optimized_dag, c.name) == (3,)

    result = c.compute(optimize_function=opt_fn)
    assert_array_equal(result, -np.ones((3, 2)))


# merge chunks with different number of tasks (c has more tasks than d)
#
#  a   b    ->   a   b
# 1 \ / 1       3 \ / 3
#    c             d
#    | 3
#    d
#
def test_fuse_merge_chunks_binary(spec):
    a = xp.ones((3, 2), chunks=(1, 2), spec=spec)
    b = xp.ones((3, 2), chunks=(1, 2), spec=spec)
    c = xp.add(a, b)
    d = merge_chunks_new(c, chunks=(3, 2))

    # specify max_total_num_input_blocks to force d to fuse
    opt_fn = fuse_multiple_levels(max_total_num_input_blocks=6)

    d.visualize(optimize_function=opt_fn)

    # check structure of optimized dag
    expected_fused_dag = create_dag()
    add_placeholder_op(expected_fused_dag, (), (a,))
    add_placeholder_op(expected_fused_dag, (), (b,))
    add_placeholder_op(expected_fused_dag, (a, b), (d,))
    optimized_dag = d.plan.optimize(optimize_function=opt_fn).dag
    assert structurally_equivalent(optimized_dag, expected_fused_dag)
    assert get_num_input_blocks(c.plan.dag, c.name) == (1, 1)
    assert get_num_input_blocks(d.plan.dag, d.name) == (3,)
    assert get_num_input_blocks(optimized_dag, d.name) == (3, 3)

    result = d.compute(optimize_function=opt_fn)
    assert_array_equal(result, 2 * np.ones((3, 2)))


# like test_fuse_merge_chunks_unary, except uses partial_reduce
def test_fuse_partial_reduce_unary(spec):
    a = xp.ones((3, 2), chunks=(1, 2), spec=spec)
    b = xp.negative(a)
    c = partial_reduce(b, np.sum, split_every={0: 3})

    # specify max_total_num_input_blocks to force c to fuse
    opt_fn = fuse_multiple_levels(max_total_num_input_blocks=3)

    c.visualize(optimize_function=opt_fn)

    # check structure of optimized dag
    expected_fused_dag = create_dag()
    add_placeholder_op(expected_fused_dag, (), (a,))
    add_placeholder_op(expected_fused_dag, (a,), (c,))
    optimized_dag = c.plan.optimize(optimize_function=opt_fn).dag
    assert structurally_equivalent(optimized_dag, expected_fused_dag)
    assert get_num_input_blocks(b.plan.dag, b.name) == (1,)
    assert get_num_input_blocks(c.plan.dag, c.name) == (3,)
    assert get_num_input_blocks(optimized_dag, c.name) == (3,)

    result = c.compute(optimize_function=opt_fn)
    assert_array_equal(result, -3 * np.ones((1, 2)))


# like test_fuse_merge_chunks_binary, except uses partial_reduce
def test_fuse_partial_reduce_binary(spec):
    a = xp.ones((3, 2), chunks=(1, 2), spec=spec)
    b = xp.ones((3, 2), chunks=(1, 2), spec=spec)
    c = xp.add(a, b)
    d = partial_reduce(c, np.sum, split_every={0: 3})

    # specify max_total_num_input_blocks to force d to fuse
    opt_fn = fuse_multiple_levels(max_total_num_input_blocks=6)

    d.visualize(optimize_function=opt_fn)

    # check structure of optimized dag
    expected_fused_dag = create_dag()
    add_placeholder_op(expected_fused_dag, (), (a,))
    add_placeholder_op(expected_fused_dag, (), (b,))
    add_placeholder_op(expected_fused_dag, (a, b), (d,))
    optimized_dag = d.plan.optimize(optimize_function=opt_fn).dag
    assert structurally_equivalent(optimized_dag, expected_fused_dag)
    assert get_num_input_blocks(c.plan.dag, c.name) == (1, 1)
    assert get_num_input_blocks(d.plan.dag, d.name) == (3,)
    assert get_num_input_blocks(optimized_dag, d.name) == (3, 3)

    result = d.compute(optimize_function=opt_fn)
    assert_array_equal(result, 6 * np.ones((1, 2)))


def test_fuse_only_optimize_dag(spec):
    a = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    b = xp.negative(a)
    c = xp.negative(b)
    d = xp.negative(c)

    # only fuse d (with c)
    # b should remain un-fused, even though it is fusable
    op_name = next(d.plan.dag.predecessors(d.name))
    opt_fn = partial(fuse_only_optimize_dag, only_fuse=[op_name])

    c.visualize(optimize_function=opt_fn)

    # check structure of optimized dag
    expected_fused_dag = create_dag()
    add_placeholder_op(expected_fused_dag, (), (a,))
    add_placeholder_op(expected_fused_dag, (a,), (b,))
    add_placeholder_op(expected_fused_dag, (b,), (d,))
    optimized_dag = d.plan.optimize(optimize_function=opt_fn).dag
    assert structurally_equivalent(optimized_dag, expected_fused_dag)
    assert get_num_input_blocks(d.plan.dag, d.name) == (1,)
    assert get_num_input_blocks(optimized_dag, d.name) == (1,)

    result = d.compute(optimize_function=opt_fn)
    assert_array_equal(result, -np.ones((2, 2)))


def test_optimize_stack(spec):
    # This test fails if stack's general_blockwise call doesn't have fusable=False
    a = cubed.random.random((10, 10), chunks=(5, 5), spec=spec)
    b = cubed.random.random((10, 10), chunks=(5, 5), spec=spec)
    c = xp.stack((a, b), axis=0)
    # try to fuse all ops into one
    c.compute(optimize_function=fuse_multiple_levels(max_total_num_input_blocks=10))
