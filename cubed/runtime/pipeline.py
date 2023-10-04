from typing import Any, Dict, Optional

import networkx as nx

from cubed.storage.zarr import open_if_lazy_zarr_array


def already_computed(node_dict: Dict[str, Any], resume: Optional[bool] = None) -> bool:
    """
    Return True if the array for a node doesn't have a pipeline to compute it,
    or it has already been computed (all chunks are present).
    """
    pipeline = node_dict.get("pipeline", None)
    if pipeline is None:
        return True

    target = node_dict.get("target", None)
    if resume and target is not None:
        target = open_if_lazy_zarr_array(target)
        # this check can be expensive since it has to list the directory to find nchunks_initialized
        if target.ndim > 0 and target.nchunks_initialized == target.nchunks:
            return True

    return False


def visit_nodes(dag, resume=None):
    """Return a generator that visits the nodes in the DAG in topological order."""
    nodes = {n: d for (n, d) in dag.nodes(data=True)}
    for name in list(nx.topological_sort(dag)):
        if already_computed(nodes[name], resume=resume):
            continue
        yield name, nodes[name]


def visit_node_generations(dag, resume=None):
    """Return a generator that visits the nodes in the DAG in groups of topological generations."""
    nodes = {n: d for (n, d) in dag.nodes(data=True)}
    for names in nx.topological_generations(dag):
        gen = [
            (name, nodes[name])
            for name in names
            if not already_computed(nodes[name], resume=resume)
        ]
        if len(gen) > 0:
            yield gen
