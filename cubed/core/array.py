import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from operator import mul

import fsspec
import networkx as nx
import numpy as np
import zarr
from dask.array.core import normalize_chunks
from dask.utils import memory_repr
from rechunker.executors.python import PythonPipelineExecutor
from rechunker.types import PipelineExecutor
from toolz import map, reduce

from cubed.rechunker_extensions.types import Executor
from cubed.utils import join_path

# A unique ID with sensible ordering, used for making directory names
CONTEXT_ID = f"context-{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4()}"

sym_counter = 0


def gensym(name="array"):
    """Generate a name with an incrementing counter"""
    global sym_counter
    sym_counter += 1
    return f"{name}-{sym_counter:03}"


class Array:
    """Chunked array backed by Zarr storage."""

    def __init__(self, name, zarray, plan):
        self.name = name
        self.zarray = zarray
        self.shape = zarray.shape
        self.dtype = zarray.dtype
        self.chunks = normalize_chunks(
            zarray.chunks, shape=self.shape, dtype=self.dtype
        )
        self.plan = plan

    def __array__(self, dtype=None):
        x = self.compute()
        if dtype and x.dtype != dtype:
            x = x.astype(dtype)
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return x

    @property
    def chunksize(self):
        return tuple(max(c) for c in self.chunks)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def numblocks(self):
        return tuple(map(len, self.chunks))

    @property
    def npartitions(self):
        return reduce(mul, self.numblocks, 1)

    @property
    def size(self):
        return reduce(mul, self.shape, 1)

    def compute(self, *, return_stored=True, executor=None, **kwargs):
        self.plan.execute(self.name, executor=executor, **kwargs)

        if return_stored:
            # read back from zarr
            return self.zarray[...]

    def visualize(self, filename="cubed", format=None):
        return self.plan.visualize(filename=filename, format=format)

    def __bool__(self, /):
        if self.ndim != 0:
            raise TypeError("bool is only allowed on arrays with 0 dimensions")
        return bool(self.compute())

    def __repr__(self):
        return f"Array<{self.name}, shape={self.shape}, dtype={self.dtype}, chunks={self.chunks}>"


class Plan:
    """Deferred computation plan for a graph of arrays.

    A thin wrapper around a NetworkX `DiGraph`. Nodes are Zarr paths, and may
    have a `pipeline` attribute holding the pipeline to execute to generate
    the output at that path. If there is no `pipeline` attribute no computation
    is needed since the Zarr file already exists. Directed edges point towards
    the dependent Zarr path.
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
            dag = nx.DiGraph(spec=spec)
        else:
            # TODO: check specs are the same, rather than just inheriting last one
            dag = nx.compose_all([x.plan.dag for x in source_arrays])

        # add new node and edges
        label = f"{name} ({op_name})"
        tooltip = (
            f"shape: {target.shape}\n"
            f"chunks: {target.chunks}\n"
            f"dtype: {target.dtype}"
        )
        if required_mem is not None:
            tooltip += f"\nmemory: {memory_repr(required_mem)}"
        if num_tasks is not None:
            tooltip += f"\ntasks: {num_tasks}"
        if pipeline is None:
            dag.add_node(name, label=label, tooltip=tooltip)
        else:
            dag.add_node(name, label=label, tooltip=tooltip, pipeline=pipeline)
        for x in source_arrays:
            dag.add_edge(name, x.name)

        self.dag = dag

    @property
    def spec(self):
        return self.dag.graph["spec"]

    def execute(self, name=None, executor=None, **kwargs):
        if executor is None:
            executor = self.spec.executor
            if executor is None:
                executor = PythonPipelineExecutor()

        # prune DAG so it only has transitive dependencies of 'name'
        dag = self.dag.subgraph(get_weakly_cc(self.dag, name))
        dag = dag.copy()

        if isinstance(executor, PipelineExecutor):

            while len(dag) > 0:
                # Find nodes (and their pipelines) that have no dependencies
                no_dep_nodes = [x for x in dag.nodes() if dag.out_degree(x) == 0]
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
            executor.execute_dag(dag, **kwargs)

    def visualize(self, filename="cubed", format=None, rankdir="BT"):
        dag = self.dag.copy()
        dag.graph["rankdir"] = rankdir
        for (_, d) in dag.nodes(data=True):
            if "pipeline" in d:
                del d["pipeline"]
        gv = nx.nx_agraph.to_agraph(dag)
        gv.node_attr["shape"] = "box"
        gv.node_attr["fontname"] = "helvetica"
        if format is None:
            format = "svg"
        full_filename = f"{filename}.{format}"
        if format == "dot":
            gv.write(full_filename)
        else:
            gv.draw(full_filename, prog="dot")

        try:
            import IPython.display as display

            if format == "svg":
                return display.SVG(filename=full_filename)
        except ImportError:
            # Can't return a display object if no IPython.
            pass
        return None


def get_weakly_cc(G, node):
    """get weakly connected component of node"""
    for cc in nx.weakly_connected_components(G):
        if node in cc:
            return cc
    else:
        return set()


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
