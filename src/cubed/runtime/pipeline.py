from typing import Any, Dict

import networkx as nx


def skip_node(name, dag, nodes: Dict[str, Any]) -> bool:
    """
    Return True if the array for a node doesn't have a pipeline to compute it,
    or if it is marked as already computed.
    """
    pipeline = nodes[name].get("pipeline", None)
    if pipeline is None:
        return True

    return nodes[name].get("computed", False)


def visit_nodes(dag):
    """Return a generator that visits the nodes in the DAG in topological order."""
    nodes = {n: d for (n, d) in dag.nodes(data=True)}
    for name in list(nx.topological_sort(dag)):
        if skip_node(name, dag, nodes):
            continue
        yield name, nodes[name]


def visit_node_generations(dag):
    """Return a generator that visits the nodes in the DAG in groups of topological generations."""
    nodes = {n: d for (n, d) in dag.nodes(data=True)}
    for names in nx.topological_generations(dag):
        gen = [(name, nodes[name]) for name in names if not skip_node(name, dag, nodes)]
        if len(gen) > 0:
            yield gen
