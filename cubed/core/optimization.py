from cubed.primitive.blockwise import can_fuse_pipelines, fuse


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
