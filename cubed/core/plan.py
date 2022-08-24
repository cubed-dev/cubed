import inspect
import sysconfig
import tempfile
import traceback
import uuid
from datetime import datetime

import fsspec
import networkx as nx
import zarr
from rechunker.types import PipelineExecutor

from cubed.primitive.blockwise import can_fuse_pipelines, fuse
from cubed.runtime.pipeline import already_computed
from cubed.utils import chunk_memory, join_path, memory_repr

# A unique ID with sensible ordering, used for making directory names
CONTEXT_ID = f"context-{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4()}"


class Plan:
    """Deferred computation plan for a graph of arrays.

    A thin wrapper around a NetworkX `MultiDiGraph`. Nodes are Zarr paths, and may
    have a `pipeline` attribute holding the pipeline to execute to generate
    the output at that path. If there is no `pipeline` attribute no computation
    is needed since the Zarr file already exists. Directed edges point towards
    the dependent Zarr path.

    Multiple edges are possible since a node (Zarr file) may be computed by
    a function with repeated inputs. For example, consider `equals` where the
    two arguments are the same array. We need to keep track of these cases, so
    we use a NetworkX `MultiDiGraph` rather than just as `DiGraph`.
    """

    # args from pipeline onwards are omitted for creation functions when no computation is needed
    def __init__(
        self,
        name,
        op_name,
        target,
        pipeline=None,
        required_mem=None,
        num_tasks=None,
        *source_arrays,
    ):
        # create an empty DAG or combine from sources
        if len(source_arrays) == 0:
            dag = nx.MultiDiGraph()
        else:
            dag = arrays_to_dag(*source_arrays)

        # add new node and edges
        frame = inspect.currentframe().f_back  # go back one in the stack
        stack_summaries = traceback.extract_stack(frame, 10)

        if pipeline is None:
            dag.add_node(
                name, op_name=op_name, target=target, stack_summaries=stack_summaries
            )
        else:
            dag.add_node(
                name,
                op_name=op_name,
                target=target,
                stack_summaries=stack_summaries,
                pipeline=pipeline,
                required_mem=required_mem,
                num_tasks=num_tasks,
            )
        for x in source_arrays:
            if hasattr(x, "name"):
                dag.add_edge(x.name, name)

        self.dag = dag


def arrays_to_dag(*arrays):
    from .array import check_array_specs

    check_array_specs(arrays)
    dags = [x.plan.dag for x in arrays if hasattr(x, "plan")]
    return nx.compose_all(dags)


def num_tasks(dag, optimize_graph=True):
    """Return the number of tasks needed to execute this dag."""
    dag = optimize_dag(dag) if optimize_graph else dag.copy()
    tasks = 0
    for _, node in visit_nodes(dag):
        pipeline = node["pipeline"]
        for stage in pipeline.stages:
            if stage.mappable is not None:
                tasks += len(list(stage.mappable))
            else:
                tasks += 1
    return tasks


def execute_dag(dag, executor=None, callbacks=None, optimize_graph=True, **kwargs):
    dag = optimize_dag(dag) if optimize_graph else dag.copy()

    if isinstance(executor, PipelineExecutor):
        while len(dag) > 0:
            # Find nodes (and their pipelines) that have no dependencies
            no_dep_nodes = [x for x in dag.nodes() if dag.in_degree(x) == 0]
            pipelines = [
                p
                for (n, p) in nx.get_node_attributes(dag, "pipeline").items()
                if n in no_dep_nodes
            ]

            # and execute them in parallel
            if len(pipelines) > 0:
                plan = executor.pipelines_to_plan(pipelines)
                executor.execute_plan(plan, **kwargs)

            # Remove them from the DAG, and repeat
            dag.remove_nodes_from(no_dep_nodes)

    else:
        if callbacks is not None:
            [callback.on_compute_start(dag) for callback in callbacks]
        executor.execute_dag(dag, callbacks=callbacks, **kwargs)
        if callbacks is not None:
            [callback.on_compute_end(dag) for callback in callbacks]


def optimize_dag(dag):
    # note there is no need to prune the dag, since the way it is built
    # ensures that only the transitive dependencies of the target arrays are included

    # fuse map blocks
    dag = dag.copy()
    nodes = {n: d for (n, d) in dag.nodes(data=True)}

    def can_fuse(n):
        # node must have a single predecessor
        #   - not multiple edges pointing to a single predecessor
        # node must be the single successor to the predecessor
        # and both must have pipelines that can be fused
        if dag.in_degree(n) != 1:
            return False
        pre = next(dag.predecessors(n))
        if dag.out_degree(pre) != 1:
            return False
        return can_fuse_pipelines(nodes[pre], nodes[n])

    for n in list(dag.nodes()):
        if can_fuse(n):
            pre = next(dag.predecessors(n))
            n1_dict = nodes[pre]
            n2_dict = nodes[n]
            pipeline, target, required_mem, num_tasks = fuse(
                n1_dict["pipeline"],
                n1_dict["target"],
                n1_dict["required_mem"],
                n1_dict["num_tasks"],
                n2_dict["pipeline"],
                n2_dict["target"],
                n2_dict["required_mem"],
                n2_dict["num_tasks"],
            )
            nodes[n]["pipeline"] = pipeline
            nodes[n]["target"] = target
            nodes[n]["required_mem"] = required_mem
            nodes[n]["num_tasks"] = num_tasks

            for p in dag.predecessors(pre):
                dag.add_edge(p, n)
            dag.remove_node(pre)

    return dag


def visualize_dag(
    dag, filename="cubed", format=None, rankdir="TB", optimize_graph=True
):
    dag = optimize_dag(dag) if optimize_graph else dag.copy()
    dag.graph["graph"] = {"rankdir": rankdir}
    dag.graph["node"] = {"fontname": "helvetica", "shape": "box"}
    for (n, d) in dag.nodes(data=True):
        d["label"] = f"{n} ({d['op_name']})"
        target = d["target"]
        chunkmem = memory_repr(chunk_memory(target.dtype, target.chunks))
        tooltip = (
            f"shape: {target.shape}\n"
            f"chunks: {target.chunks}\n"
            f"dtype: {target.dtype}\n"
            f"chunk memory: {chunkmem}\n"
        )
        if "required_mem" in d:
            tooltip += f"\ntask memory: {memory_repr(d['required_mem'])}"
        if "num_tasks" in d:
            tooltip += f"\ntasks: {d['num_tasks']}"
        if "stack_summaries" in d:
            # add call stack information
            stack_summaries = d["stack_summaries"]
            python_lib_path = sysconfig.get_path("purelib")
            calls = " -> ".join(
                [
                    s.name
                    for s in stack_summaries
                    if not s.filename.startswith(python_lib_path)
                ]
            )
            tooltip += f"\ncalls: {calls}"
            del d["stack_summaries"]

        d["tooltip"] = tooltip

        # remove pipeline attribute since it is a long string that causes graphviz to fail
        if "pipeline" in d:
            del d["pipeline"]
        if "target" in d:
            del d["target"]
    gv = nx.drawing.nx_pydot.to_pydot(dag)
    if format is None:
        format = "svg"
    full_filename = f"{filename}.{format}"
    gv.write(full_filename, format=format)

    try:  # pragma: no cover
        import IPython.display as display

        if format == "svg":
            return display.SVG(filename=full_filename)
    except ImportError:
        # Can't return a display object if no IPython.
        pass
    return None


def new_temp_path(name, suffix, spec=None):
    work_dir = spec.work_dir if spec is not None else None
    if work_dir is None:
        work_dir = tempfile.gettempdir()
    context_dir = join_path(work_dir, CONTEXT_ID)
    return join_path(context_dir, f"{name}{suffix}")


def new_temp_store(name, spec=None):
    zarr_path = new_temp_path(name, ".zarr", spec)
    return fsspec.get_mapper(zarr_path)


def new_temp_zarr(shape, dtype, chunksize, name=None, spec=None):
    # open a new temporary zarr array for writing
    store = new_temp_store(name=name, spec=spec)
    target = zarr.open(store, mode="w-", shape=shape, dtype=dtype, chunks=chunksize)
    return target


def visit_nodes(dag):
    """Return a generator that visits the nodes in the DAG in topological order."""
    nodes = {n: d for (n, d) in dag.nodes(data=True)}
    for name in list(nx.topological_sort(dag)):
        if already_computed(nodes[name]):
            continue
        yield name, nodes[name]


def visit_node_generations(dag):
    """Return a generator that visits the nodes in the DAG in groups of topological generations."""
    nodes = {n: d for (n, d) in dag.nodes(data=True)}
    for names in nx.topological_generations(dag):
        gen = [
            (name, nodes[name]) for name in names if not already_computed(nodes[name])
        ]
        if len(gen) > 0:
            yield gen
