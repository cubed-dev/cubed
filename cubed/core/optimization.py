import networkx as nx

from cubed.primitive.blockwise import (
    can_fuse_multiple_pipelines,
    can_fuse_pipelines,
    fuse,
    fuse_multiple,
)


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

        # if node (op2) does not have a pipeline then it can't be fused
        if "pipeline" not in nodes[op2]:
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

        # op1 and op2 must have pipelines that can be fused
        if "pipeline" not in nodes[op1]:
            return False
        return can_fuse_pipelines(nodes[op1]["pipeline"], nodes[op2]["pipeline"])

    for n in list(dag.nodes()):
        if can_fuse(n):
            op2 = n
            op2_input = next(dag.predecessors(op2))
            op1 = next(dag.predecessors(op2_input))
            op1_inputs = list(dag.predecessors(op1))

            pipeline = fuse(nodes[op1]["pipeline"], nodes[op2]["pipeline"])
            nodes[op2]["pipeline"] = pipeline

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


def predecessors(dag, name):
    """Return a node's predecessors, with repeats for multiple edges."""
    for pre, _ in dag.in_edges(name):
        yield pre


def predecessor_ops(dag, name):
    """Return an op node's op predecessors"""
    for input in predecessors(dag, name):
        for pre in predecessors(dag, input):
            yield pre


def is_fusable(node_dict):
    "Return True if a node can be fused."
    return "pipeline" in node_dict


def can_fuse_predecessors(dag, name, *, max_total_nargs=4):
    nodes = dict(dag.nodes(data=True))

    # if node itself can't be fused then there is nothing to fuse
    if not is_fusable(nodes[name]):
        return False

    # if no predecessor ops can be fused then there is nothing to fuse
    if all(not is_fusable(nodes[pre]) for pre in predecessor_ops(dag, name)):
        return False

    # if there is more than a single predecessor op, and the total number of args to
    # the fused function would be more than an allowed maximum, then don't fuse
    if len(list(predecessor_ops(dag, name))) > 1:
        total_nargs = sum(
            len(list(predecessors(dag, pre))) if is_fusable(nodes[pre]) else 1
            for pre in predecessor_ops(dag, name)
        )
        if total_nargs > max_total_nargs:
            return False

    predecessor_pipelines = [
        nodes[pre]["pipeline"]
        for pre in predecessor_ops(dag, name)
        if is_fusable(nodes[pre])
    ]
    return can_fuse_multiple_pipelines(nodes[name]["pipeline"], *predecessor_pipelines)


def fuse_predecessors(dag, name):
    """Fuse a node with its immediate predecessors."""

    # if can't fuse then return dag unchanged
    if not can_fuse_predecessors(dag, name):
        return dag

    nodes = dict(dag.nodes(data=True))

    pipeline = nodes[name]["pipeline"]

    # if a predecessor op has no pipeline then just use None
    predecessor_pipelines = [
        nodes[pre]["pipeline"] if is_fusable(nodes[pre]) else None
        for pre in predecessor_ops(dag, name)
    ]

    # if a predecessor op has no func then use 1 for nargs
    predecessor_funcs_nargs = [
        len(list(predecessors(dag, pre))) if is_fusable(nodes[pre]) else 1
        for pre in predecessor_ops(dag, name)
    ]

    fused_pipeline = fuse_multiple(
        pipeline,
        *predecessor_pipelines,
        predecessor_funcs_nargs=predecessor_funcs_nargs,
    )

    fused_dag = dag.copy()
    fused_nodes = dict(fused_dag.nodes(data=True))

    fused_nodes[name]["pipeline"] = fused_pipeline

    # re-wire dag to remove predecessor nodes that have been fused

    # 1. update edges to change inputs
    for input in predecessors(dag, name):
        pre = next(predecessors(dag, input))
        if not is_fusable(fused_nodes[pre]):
            # if a predecessor is marked as not fusable then don't change the edge
            continue
        fused_dag.remove_edge(input, name)
    for pre in predecessor_ops(dag, name):
        if not is_fusable(fused_nodes[pre]):
            # if a predecessor is marked as not fusable then don't change the edge
            continue
        for input in predecessors(dag, pre):
            fused_dag.add_edge(input, name)

    # 2. remove predecessor nodes with no successors
    # (ones with successors are needed by other nodes)
    for input in predecessors(dag, name):
        if fused_dag.out_degree(input) == 0:
            for pre in list(predecessors(fused_dag, input)):
                fused_dag.remove_node(pre)
            fused_dag.remove_node(input)

    return fused_dag


def multiple_inputs_optimize_dag(dag):
    """Fuse multiple inputs."""
    for name in list(nx.topological_sort(dag)):
        dag = fuse_predecessors(dag, name)
    return dag
