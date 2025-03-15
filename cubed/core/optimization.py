import logging

import networkx as nx

from cubed.primitive.blockwise import (
    can_fuse_multiple_primitive_ops,
    can_fuse_primitive_ops,
    fuse,
    fuse_multiple,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOTAL_SOURCE_ARRAYS = 4
DEFAULT_MAX_TOTAL_NUM_INPUT_BLOCKS = 10


def simple_optimize_dag(dag, array_names=None):
    """Apply map blocks fusion."""

    # note there is no need to prune the dag, since the way it is built
    # ensures that only the transitive dependencies of the target arrays are included

    dag = dag.copy()
    nodes = {n: d for (n, d) in dag.nodes(data=True)}

    def can_fuse(n):
        # fuse a single chain looking like this:
        # op1 -> op2_input -> op2

        op2 = n

        # if node (op2) does not have a primitive op then it can't be fused
        if "primitive_op" not in nodes[op2]:
            return False

        # if node (op2) does not have exactly one input and output then don't fuse
        # (it could have no inputs or multiple inputs)
        if dag.in_degree(op2) != 1 or dag.out_degree(op2) != 1:
            return False

        # if input is one of the arrays being computed then don't fuse
        op2_input = next(dag.predecessors(op2))
        if array_names is not None and op2_input in array_names:
            return False

        # if input is used by another node then don't fuse
        if dag.out_degree(op2_input) != 1:
            return False

        # if node producing input (op1) has more than one output then don't fuse
        op1 = next(dag.predecessors(op2_input))
        if dag.out_degree(op1) != 1:
            return False

        # op1 and op2 must have primitive ops that can be fused
        if "primitive_op" not in nodes[op1]:
            return False
        return can_fuse_primitive_ops(
            nodes[op1]["primitive_op"], nodes[op2]["primitive_op"]
        )

    for n in list(dag.nodes()):
        if can_fuse(n):
            op2 = n
            op2_input = next(dag.predecessors(op2))
            op1 = next(dag.predecessors(op2_input))
            op1_inputs = list(dag.predecessors(op1))

            primitive_op = fuse(nodes[op1]["primitive_op"], nodes[op2]["primitive_op"])
            nodes[op2]["primitive_op"] = primitive_op
            nodes[op2]["pipeline"] = primitive_op.pipeline

            for n in op1_inputs:
                dag.add_edge(n, op2)
            dag.remove_node(op2_input)
            dag.remove_node(op1)

    return dag


sym_counter = 0


def gensym(name="op"):
    global sym_counter
    sym_counter += 1
    return f"{name}-{sym_counter:03}"


def predecessors_unordered(dag, name):
    """Return a node's predecessors in no particular order, with repeats for multiple edges."""
    for pre, _ in dag.in_edges(name):
        yield pre


def successors_unordered(dag, name):
    """Return a node's successors in no particular order, with repeats for multiple edges."""
    for pre, _ in dag.out_edges(name):
        yield pre


def predecessor_ops(dag, name):
    """Return an op node's op predecessors in the same order as the input source arrays for the op.

    Note that each input source array is produced by a single predecessor op.
    """
    nodes = dict(dag.nodes(data=True))
    for input in nodes[name]["primitive_op"].source_array_names:
        pre_list = list(predecessors_unordered(dag, input))
        assert len(pre_list) == 1  # each array is produced by a single op
        yield pre_list[0]


def predecessor_ops_and_arrays(dag, name):
    # returns op predecessors, the arrays that they produce (only one since we don't support multiple outputs yet),
    # and a flag indicating if the op can be fused with each predecessor, taking into account the number of dependents for the array
    nodes = dict(dag.nodes(data=True))
    for input in nodes[name]["primitive_op"].source_array_names:
        pre_list = list(predecessors_unordered(dag, input))
        assert len(pre_list) == 1  # each array is produced by a single op
        pre = pre_list[0]
        can_fuse = is_primitive_op(nodes[pre]) and out_degree_unique(dag, input) == 1
        yield pre, input, can_fuse


def out_degree_unique(dag, name):
    """Returns number of unique out edges"""
    return len(set(post for _, post in dag.out_edges(name)))


def is_primitive_op(node_dict):
    """Return True if a node is a primitive op"""
    return "primitive_op" in node_dict


def is_fusable_with_predecessors(node_dict):
    """Return True if a node is a primitive op and can be fused with its predecessors."""
    return (
        is_primitive_op(node_dict)
        and node_dict["primitive_op"].fusable_with_predecessors
    )


def num_source_arrays(dag, name):
    """Return the number of (non-hidden) arrays that are inputs to an op.

    Hidden arrays are used for internal bookkeeping, are very small virtual arrays
    (empty, or offsets for example), and are not shown on the plan visualization.
    For these reasons they shouldn't count towards ``max_total_source_arrays``.
    """
    nodes = dict(dag.nodes(data=True))
    return sum(
        not nodes[array]["hidden"] for array in predecessors_unordered(dag, name)
    )


def can_fuse_predecessors(
    dag,
    name,
    *,
    array_names=None,
    max_total_source_arrays=DEFAULT_MAX_TOTAL_SOURCE_ARRAYS,
    max_total_num_input_blocks=DEFAULT_MAX_TOTAL_NUM_INPUT_BLOCKS,
    always_fuse=None,
    never_fuse=None,
):
    nodes = dict(dag.nodes(data=True))

    # if node itself can't be fused then there is nothing to fuse
    if not is_fusable_with_predecessors(nodes[name]):
        logger.debug(
            "can't fuse %s since it is not a primitive operation, or it uses an operation that can't be fused (concat or stack)",
            name,
        )
        return False

    # if no predecessor ops can be fused then there is nothing to fuse
    # (this may be because predecessor ops produce arrays with multiple dependents)
    if all(not can_fuse for _, _, can_fuse in predecessor_ops_and_arrays(dag, name)):
        logger.debug("can't fuse %s since no predecessor ops can be fused", name)
        return False

    # if a predecessor op produces one of the arrays being computed, then don't fuse
    if array_names is not None:
        predecessor_array_names = set(
            array_name for _, array_name, _ in predecessor_ops_and_arrays(dag, name)
        )
        array_names_intersect = set(array_names) & predecessor_array_names
        if len(array_names_intersect) > 0:
            logger.debug(
                "can't fuse %s since predecessor ops produce one or more arrays being computed %s",
                name,
                array_names_intersect,
            )
            return False

    # if any predecessor ops have multiple outputs then don't fuse
    # TODO: implement "child fusion" (where a multiple output op fuses its children)
    if any(
        len(list(successors_unordered(dag, pre))) > 1
        for pre in predecessor_ops(dag, name)
    ):
        logger.debug(
            "can't fuse %s since at least one predecessor has multiple outputs", name
        )
        return False

    # if node is in never_fuse or always_fuse list then it overrides logic below
    if never_fuse is not None and name in never_fuse:
        logger.debug("can't fuse %s since it is in 'never_fuse'", name)
        return False
    if always_fuse is not None and name in always_fuse:
        logger.debug("can fuse %s since it is in 'always_fuse'", name)
        return True

    # if there is more than a single predecessor op, and the total number of source arrays to
    # the fused function would be more than an allowed maximum, then don't fuse
    if len(list(predecessor_ops(dag, name))) > 1:
        total_source_arrays = sum(
            num_source_arrays(dag, pre) if can_fuse else 1
            for pre, _, can_fuse in predecessor_ops_and_arrays(dag, name)
        )
        if total_source_arrays > max_total_source_arrays:
            logger.debug(
                "can't fuse %s since total number of source arrays (%s) exceeds max (%s)",
                name,
                total_source_arrays,
                max_total_source_arrays,
            )
            return False

    # if a predecessor has no primitive op then just use None
    predecessor_primitive_ops = [
        nodes[pre]["primitive_op"] if can_fuse else None
        for pre, _, can_fuse in predecessor_ops_and_arrays(dag, name)
    ]
    return can_fuse_multiple_primitive_ops(
        name,
        nodes[name]["primitive_op"],
        predecessor_primitive_ops,
        max_total_num_input_blocks=max_total_num_input_blocks,
    )


def fuse_predecessors(
    dag,
    name,
    *,
    array_names=None,
    max_total_source_arrays=DEFAULT_MAX_TOTAL_SOURCE_ARRAYS,
    max_total_num_input_blocks=DEFAULT_MAX_TOTAL_NUM_INPUT_BLOCKS,
    always_fuse=None,
    never_fuse=None,
):
    """Fuse a node with its immediate predecessors."""

    # if can't fuse then return dag unchanged
    if not can_fuse_predecessors(
        dag,
        name,
        array_names=array_names,
        max_total_source_arrays=max_total_source_arrays,
        max_total_num_input_blocks=max_total_num_input_blocks,
        always_fuse=always_fuse,
        never_fuse=never_fuse,
    ):
        return dag

    nodes = dict(dag.nodes(data=True))

    primitive_op = nodes[name]["primitive_op"]

    # if a predecessor has no primitive op then just use None
    predecessor_primitive_ops = [
        nodes[pre]["primitive_op"] if can_fuse else None
        for pre, _, can_fuse in predecessor_ops_and_arrays(dag, name)
    ]

    fused_primitive_op = fuse_multiple(primitive_op, *predecessor_primitive_ops)

    fused_dag = dag.copy()
    fused_nodes = dict(fused_dag.nodes(data=True))

    fused_nodes[name]["primitive_op"] = fused_primitive_op
    fused_nodes[name]["pipeline"] = fused_primitive_op.pipeline

    # re-wire dag to remove predecessor nodes that have been fused
    for pre, input, can_fuse in predecessor_ops_and_arrays(dag, name):
        if can_fuse:
            # check if already removed for repeated arguments
            if input in fused_dag:
                fused_dag.remove_node(input)
            if pre in fused_dag:
                fused_dag.remove_node(pre)
            for pre_input in predecessors_unordered(dag, pre):
                fused_dag.add_edge(pre_input, name)

    return fused_dag


def multiple_inputs_optimize_dag(
    dag,
    *,
    array_names=None,
    max_total_source_arrays=DEFAULT_MAX_TOTAL_SOURCE_ARRAYS,
    max_total_num_input_blocks=DEFAULT_MAX_TOTAL_NUM_INPUT_BLOCKS,
    always_fuse=None,
    never_fuse=None,
):
    """Fuse multiple inputs."""
    for name in list(nx.topological_sort(dag)):
        if name.startswith("array-"):
            continue
        dag = fuse_predecessors(
            dag,
            name,
            array_names=array_names,
            max_total_source_arrays=max_total_source_arrays,
            max_total_num_input_blocks=max_total_num_input_blocks,
            always_fuse=always_fuse,
            never_fuse=never_fuse,
        )
    return dag


def fuse_all_optimize_dag(dag, array_names=None):
    """Force all operations to be fused."""
    dag = dag.copy()
    always_fuse = [op for op in dag.nodes() if op.startswith("op-")]
    return multiple_inputs_optimize_dag(
        dag, array_names=array_names, always_fuse=always_fuse
    )


def fuse_only_optimize_dag(dag, *, array_names=None, only_fuse=None):
    """Force only specified operations to be fused, all others will be left even if they are suitable for fusion."""
    dag = dag.copy()
    always_fuse = only_fuse
    never_fuse = set(op for op in dag.nodes() if op.startswith("op-")) - set(only_fuse)
    return multiple_inputs_optimize_dag(
        dag, array_names=array_names, always_fuse=always_fuse, never_fuse=never_fuse
    )
