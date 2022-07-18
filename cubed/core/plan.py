import inspect
import sysconfig
import tempfile
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime

import fsspec
import networkx as nx
import zarr
from rechunker.executors.python import PythonPipelineExecutor
from rechunker.types import PipelineExecutor

from cubed.primitive.blockwise import can_fuse_pipelines, fuse
from cubed.runtime.pipeline import already_computed
from cubed.runtime.types import Executor
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
        spec,
        pipeline=None,
        required_mem=None,
        num_tasks=None,
        *source_arrays,
    ):
        # if no spec is supplied, use a default with local temp dir, and a modest amount of memory (100MB)
        if spec is None:
            spec = Spec(None, 100_000_000)

        # create an empty DAG or combine from sources
        if len(source_arrays) == 0:
            dag = nx.MultiDiGraph(spec=spec)
        else:
            # TODO: check specs are the same, rather than just inheriting last one
            dag = nx.compose_all([x.plan.dag for x in source_arrays])

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
            dag.add_edge(x.name, name)

        self.dag = dag

    @property
    def spec(self):
        return self.dag.graph["spec"]

    def transitive_dependencies(self, name):
        # prune DAG so it only has transitive dependencies of 'name'
        if name is None:
            return self.dag
        for cc in nx.weakly_connected_components(self.dag):
            if name in cc:
                return self.dag.subgraph(cc)
        else:
            raise ValueError(f"{name} not found")

    def num_tasks(self, name=None, optimize_graph=True):
        """Return the number of tasks needed to execute this plan."""
        tasks = 0
        if optimize_graph:
            dag = self.optimize(name)
        else:
            dag = self.dag.copy()
        nodes = {n: d for (n, d) in dag.nodes(data=True)}
        for node in dag:
            if already_computed(nodes[node]):
                continue
            pipeline = nodes[node]["pipeline"]
            for stage in pipeline.stages:
                if stage.mappable is not None:
                    tasks += len(list(stage.mappable))
                else:
                    tasks += 1
        return tasks

    def optimize(self, name=None):
        # remove anything that 'name' node doesn't depend on
        dag = self.transitive_dependencies(name)

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

    def execute(
        self, name=None, executor=None, callbacks=None, optimize_graph=True, **kwargs
    ):
        if executor is None:
            executor = self.spec.executor
            if executor is None:
                executor = PythonPipelineExecutor()

        if optimize_graph:
            dag = self.optimize(name)
        else:
            dag = self.dag.copy()

        if isinstance(executor, PipelineExecutor):
            dag = dag.copy()

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
            executor.execute_dag(dag, callbacks=callbacks, **kwargs)

    def visualize(
        self, filename="cubed", format=None, rankdir="TB", optimize_graph=True
    ):
        if optimize_graph:
            dag = self.optimize()
        else:
            dag = self.dag.copy()
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
        gv = nx.drawing.nx_pydot.to_pydot(dag)
        if format is None:
            format = "svg"
        full_filename = f"{filename}.{format}"
        gv.write(full_filename, format=format)

        try:
            import IPython.display as display

            if format == "svg":
                return display.SVG(filename=full_filename)
        except ImportError:  # pragma: no cover
            # Can't return a display object if no IPython.
            pass
        return None


@dataclass
class Spec:
    """Specification of resources available to run a computation."""

    work_dir: str
    max_mem: int
    executor: Executor = None
    storage_options: dict = None


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
