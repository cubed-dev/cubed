import logging

import networkx as nx

from cubed.primitive.blockwise import (
    can_fuse_multiple_primitive_ops,
    can_fuse_primitive_ops,
    fuse,
    fuse_multiple,
)

logger = logging.getLogger(__name__)


def simple_optimize_dag(dag):
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

        # if node (op2) does not have exactly one input then don't fuse
        # (it could have no inputs or multiple inputs)
        if dag.in_degree(op2) != 1:
            return False

        # if input is used by another node then don't fuse
        op2_input = next(dag.predecessors(op2))
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


def predecessor_ops(dag, name):
    """Return an op node's op predecessors in the same order as the input source arrays for the op.

    Note that each input source array is produced by a single predecessor op.
    """
    nodes = dict(dag.nodes(data=True))
    for input in nodes[name]["primitive_op"].source_array_names:
        pre_list = list(predecessors_unordered(dag, input))
        assert len(pre_list) == 1  # each array is produced by a single op
        yield pre_list[0]


def is_fusable(node_dict):
    "Return True if a node can be fused."
    return "primitive_op" in node_dict and node_dict["primitive_op"].fusable


def can_fuse_predecessors(
    dag,
    name,
    *,
    max_total_source_arrays=4,
    max_total_num_input_blocks=None,
    always_fuse=None,
    never_fuse=None,
):
    nodes = dict(dag.nodes(data=True))

    # if node itself can't be fused then there is nothing to fuse
    if not is_fusable(nodes[name]):
        logger.debug(
            "can't fuse %s since it is not a primitive operation, or it uses map_direct",
            name,
        )
        return False

    # if no predecessor ops can be fused then there is nothing to fuse
    if all(not is_fusable(nodes[pre]) for pre in predecessor_ops(dag, name)):
        logger.debug("can't fuse %s since no predecessor ops can be fused", name)
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
            len(list(predecessors_unordered(dag, pre))) if is_fusable(nodes[pre]) else 1
            for pre in predecessor_ops(dag, name)
        )
        if total_source_arrays > max_total_source_arrays:
            logger.debug(
                "can't fuse %s since total number of source arrays (%s) exceeds max (%s)",
                name,
                total_source_arrays,
                max_total_source_arrays,
            )
            return False

    predecessor_primitive_ops = [
        nodes[pre]["primitive_op"]
        for pre in predecessor_ops(dag, name)
        if is_fusable(nodes[pre])
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
    max_total_source_arrays=4,
    max_total_num_input_blocks=None,
    always_fuse=None,
    never_fuse=None,
):
    """Fuse a node with its immediate predecessors."""

    # if can't fuse then return dag unchanged
    if not can_fuse_predecessors(
        dag,
        name,
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
        nodes[pre]["primitive_op"] if is_fusable(nodes[pre]) else None
        for pre in predecessor_ops(dag, name)
    ]

    fused_primitive_op = fuse_multiple(primitive_op, *predecessor_primitive_ops)

    fused_dag = dag.copy()
    fused_nodes = dict(fused_dag.nodes(data=True))

    fused_nodes[name]["primitive_op"] = fused_primitive_op
    fused_nodes[name]["pipeline"] = fused_primitive_op.pipeline

    # re-wire dag to remove predecessor nodes that have been fused

    # 1. update edges to change inputs
    for input in predecessors_unordered(dag, name):
        pre = next(predecessors_unordered(dag, input))
        if not is_fusable(fused_nodes[pre]):
            # if a predecessor is not fusable then don't change the edge
            continue
        fused_dag.remove_edge(input, name)
    for pre in predecessor_ops(dag, name):
        if not is_fusable(fused_nodes[pre]):
            # if a predecessor is not fusable then don't change the edge
            continue
        for input in predecessors_unordered(dag, pre):
            fused_dag.add_edge(input, name)

    # 2. remove predecessor nodes with no successors
    # (ones with successors are needed by other nodes)
    for input in predecessors_unordered(dag, name):
        if fused_dag.out_degree(input) == 0:
            for pre in list(predecessors_unordered(fused_dag, input)):
                fused_dag.remove_node(pre)
            fused_dag.remove_node(input)

    return fused_dag


def multiple_inputs_optimize_dag(
    dag,
    *,
    max_total_source_arrays=4,
    max_total_num_input_blocks=None,
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
            max_total_source_arrays=max_total_source_arrays,
            max_total_num_input_blocks=max_total_num_input_blocks,
            always_fuse=always_fuse,
            never_fuse=never_fuse,
        )
    return dag


def fuse_all_optimize_dag(dag):
    """Force all operations to be fused."""
    dag = dag.copy()
    always_fuse = [op for op in dag.nodes() if op.startswith("op-")]
    return multiple_inputs_optimize_dag(dag, always_fuse=always_fuse)


def fuse_only_optimize_dag(dag, *, only_fuse=None):
    """Force only specified operations to be fused, all others will be left even if they are suitable for fusion."""
    dag = dag.copy()
    always_fuse = only_fuse
    never_fuse = set(op for op in dag.nodes() if op.startswith("op-")) - set(only_fuse)
    return multiple_inputs_optimize_dag(
        dag, always_fuse=always_fuse, never_fuse=never_fuse
    )
