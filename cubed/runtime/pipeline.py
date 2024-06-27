from typing import Any, Dict, Optional

import networkx as nx

from cubed.storage.zarr import open_if_lazy_zarr_array


def already_computed(
    name, dag, nodes: Dict[str, Any], resume: Optional[bool] = None
) -> bool:
    """
    Return True if the array for a node doesn't have a pipeline to compute it,
    or if all its outputs have already been computed (all chunks are present).
    """
    pipeline = nodes[name].get("pipeline", None)
    if pipeline is None:
        return True

    # if no outputs have targets then need to compute (this is the create-arrays case)
    if all(
        [nodes[output].get("target", None) is None for output in dag.successors(name)]
    ):
        return False

    if resume:
        for output in dag.successors(name):
            target = nodes[output].get("target", None)
            if target is not None:
                target = open_if_lazy_zarr_array(target)
                if not hasattr(target, "nchunks_initialized"):
                    raise NotImplementedError(
                        f"Zarr array type {type(target)} does not support resume since it doesn't have a 'nchunks_initialized' property"
                    )
                # this check can be expensive since it has to list the directory to find nchunks_initialized
                if target.ndim == 0 or target.nchunks_initialized != target.nchunks:
                    return False
        return True

    return False


def visit_nodes(dag, resume=None):
    """Return a generator that visits the nodes in the DAG in topological order."""
    nodes = {n: d for (n, d) in dag.nodes(data=True)}
    for name in list(nx.topological_sort(dag)):
        if already_computed(name, dag, nodes, resume=resume):
            continue
        yield name, nodes[name]


def visit_node_generations(dag, resume=None):
    """Return a generator that visits the nodes in the DAG in groups of topological generations."""
    nodes = {n: d for (n, d) in dag.nodes(data=True)}
    for names in nx.topological_generations(dag):
        gen = [
            (name, nodes[name])
            for name in names
            if not already_computed(name, dag, nodes, resume=resume)
        ]
        if len(gen) > 0:
            yield gen
