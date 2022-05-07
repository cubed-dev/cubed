from dataclasses import dataclass
from math import prod
from numbers import Integral, Number
from operator import mul

import fsspec
import networkx as nx
import numpy as np
import zarr
from dask.array.core import normalize_chunks
from dask.array.utils import validate_axis
from dask.utils import memory_repr
from rechunker.executors.python import PythonPipelineExecutor
from rechunker.types import PipelineExecutor
from tlz import concat
from toolz import map, reduce

from barry.primitive import blockwise as primitive_blockwise
from barry.primitive import rechunk as primitive_rechunk
from barry.rechunker_extensions.types import Executor
from barry.utils import temporary_directory

sym_counter = 0


def gensym(name="array"):
    global sym_counter
    sym_counter += 1
    return f"{name}-{sym_counter:03}"


class Array:
    """Chunked array backed by Zarr storage."""

    def __init__(self, name, plan, zarray, shape, dtype, chunks):
        self.name = name
        self.plan = plan
        self.zarray = zarray
        self.shape = shape
        self.dtype = dtype
        self.chunks = normalize_chunks(chunks, shape, dtype)

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
    def size(self):
        return reduce(mul, self.shape, 1)

    def compute(self, *, return_stored=True, executor=None, **kwargs):
        self.plan.execute(self.name, executor=executor, **kwargs)

        if return_stored:
            # read back from zarr
            return self.zarray[...]

    def visualize(self, filename="barry", format=None):
        return self.plan.visualize(filename=filename, format=format)

    def __bool__(self, /):
        if self.ndim != 0:
            raise TypeError("bool is only allowed on arrays with 0 dimensions")
        return bool(self.compute())

    def __repr__(self):
        return f"Array<{self.name}, shape={self.shape}, dtype={self.dtype}, chunks={self.chunks}>"


class Plan:
    """Deferred computation plan for a graph of arrays."""

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
        self.dag = Plan.create_dag(
            name, op_name, target, pipeline, required_mem, num_tasks, *source_arrays
        )
        # if no spec is supplied, use a default with local temp dir, and a modest amount of memory (100MB)
        self.spec = spec if spec is not None else Spec(None, 100_000_000)
        self.complete = False

    @staticmethod
    def create_dag(
        name, op_name, target, pipeline, required_mem, num_tasks, *source_arrays
    ):
        """
        Create a DAG for an Array.
        The nodes are Zarr paths, and may have 'pipeline' attribute holding the pipeline to execute to generate the output at that path.
        If there is no 'pipeline' attribute no computation is needed since the Zarr file already exists.
        Directed edges point towards the dependent path.
        """

        # copy DAGs from sources
        if len(source_arrays) == 0:
            dag = nx.DiGraph()
        else:
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

        return dag

    def execute(self, name=None, executor=None, **kwargs):
        if self.complete:
            return

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

        self.complete = True

    def visualize(self, filename="barry", format=None, rankdir="BT"):
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


def new_temp_store(name=None, spec=None):
    work_dir = spec.work_dir if spec is not None else None
    path = temporary_directory(suffix=".zarr", prefix=f"{name}-", dir=work_dir)
    return fsspec.get_mapper(path)


def new_temp_zarr(shape, dtype, chunksize, name=None, spec=None):
    # open a new temporary zarr array for writing
    store = new_temp_store(name=name, spec=spec)
    target = zarr.open(store, mode="w-", shape=shape, dtype=dtype, chunks=chunksize)
    return store, target


# General array operations


def from_zarr(store, spec=None):
    """Load an array from Zarr storage."""
    name = gensym()
    target = zarr.open(store, mode="r")

    plan = Plan(name, "from_zarr", target, spec)
    return Array(name, plan, target, target.shape, target.dtype, target.chunks)


def to_zarr(x, store, return_stored=False, executor=None):
    """Save an array to Zarr storage."""
    # Use rechunk with same chunks to implement a straight copy.
    # It would be good to avoid this copy in the future. Maybe allow optional store to be passed to all functions?
    out = rechunk(x, x.chunksize, target_store=store)
    return out.compute(return_stored=return_stored, executor=executor)


def blockwise(func, out_ind, *args, dtype=None, adjust_chunks=None, **kwargs):
    arrays = args[::2]

    assert len(arrays) > 0

    # replace arrays with zarr arrays
    zargs = list(args)
    zargs[::2] = [a.zarray for a in arrays]

    name = gensym()
    spec = arrays[0].plan.spec
    target_store = new_temp_store(name=name, spec=spec)
    pipeline, target, required_mem, num_tasks = primitive_blockwise(
        func,
        out_ind,
        *zargs,
        max_mem=spec.max_mem,
        target_store=target_store,
        dtype=dtype,
        adjust_chunks=adjust_chunks,
        **kwargs,
    )
    plan = Plan(
        name, "blockwise", target, spec, pipeline, required_mem, num_tasks, *arrays
    )
    return Array(name, plan, target, target.shape, dtype, target.chunks)


def elementwise_unary_operation(x, func, dtype):
    return map_blocks(func, x, dtype=dtype)


def elementwise_binary_operation(x1, x2, func, dtype):
    # TODO: check x1 and x2 are compatible
    # TODO: unify_chunks

    return map_blocks(func, x1, x2, dtype=dtype)


def map_blocks(func, *args, dtype=None, chunks=None, drop_axis=[], **kwargs):
    # based on dask
    if isinstance(drop_axis, Number):
        drop_axis = [drop_axis]

    arrs = args
    argpairs = [
        (a, tuple(range(a.ndim))[::-1]) if isinstance(a, Array) else (a, None)
        for a in args
    ]
    if arrs:
        out_ind = tuple(range(max(a.ndim for a in arrs)))[::-1]
    else:
        out_ind = ()

    if drop_axis:
        ndim_out = len(out_ind)
        if any(i < -ndim_out or i >= ndim_out for i in drop_axis):
            raise ValueError(
                f"drop_axis out of range (drop_axis={drop_axis}, "
                f"but output is {ndim_out}d)."
            )
        drop_axis = [i % ndim_out for i in drop_axis]
        out_ind = tuple(x for i, x in enumerate(out_ind) if i not in drop_axis)

    if chunks is not None:
        if len(chunks) != len(out_ind):
            raise ValueError(
                f"Provided chunks have {len(chunks)} dims; expected {len(out_ind)} dims"
            )
        adjust_chunks = dict(zip(out_ind, chunks))
    else:
        adjust_chunks = None

    return blockwise(
        func,
        out_ind,
        *concat(argpairs),
        dtype=dtype,
        adjust_chunks=adjust_chunks,
        **kwargs,
    )


def rechunk(x, chunks, target_store=None):
    name = gensym()
    spec = x.plan.spec
    if target_store is None:
        target_store = new_temp_store(name=name, spec=spec)
    temp_store = new_temp_store(name=f"{name}-intermediate", spec=spec)
    pipeline, target, required_mem, num_tasks = primitive_rechunk(
        x.zarray,
        target_chunks=chunks,
        max_mem=spec.max_mem,
        target_store=target_store,
        temp_store=temp_store,
    )
    plan = Plan(name, "rechunk", target, spec, pipeline, required_mem, num_tasks, x)
    return Array(name, plan, target, target.shape, target.dtype, target.chunks)


def reduction(x, func, axis=None, dtype=None, keepdims=False):
    if axis is None:
        axis = tuple(range(x.ndim))
    if isinstance(axis, Integral):
        axis = (axis,)
    axis = validate_axis(axis, x.ndim)

    inds = tuple(range(x.ndim))

    result = x
    max_mem = x.plan.spec.max_mem

    # reduce chunks (if any axis chunksize is > 1)
    if any(s > 1 for i, s in enumerate(result.chunksize) if i in axis):
        args = (result, inds)
        adjust_chunks = {
            i: (1,) * len(c) if i in axis else c for i, c in enumerate(result.chunks)
        }
        result = blockwise(
            func,
            inds,
            *args,
            axis=axis,
            keepdims=True,
            dtype=dtype,
            adjust_chunks=adjust_chunks,
        )

    # rechunk/reduce along axis in multiple rounds until there's a single block in each reduction axis
    while any(n > 1 for i, n in enumerate(result.numblocks) if i in axis):
        # rechunk along axis
        target_chunks = list(result.chunksize)
        chunk_mem = np.dtype(dtype).itemsize * prod(result.chunksize)
        for i, s in enumerate(result.shape):
            if i in axis:
                if len(axis) > 1:
                    # TODO: make sure chunk size doesn't exceed max_mem for multi-axis reduction
                    target_chunks[i] = s
                else:
                    target_chunks[i] = min(s, (max_mem - chunk_mem) // chunk_mem)
        target_chunks = tuple(target_chunks)
        result = rechunk(result, target_chunks)

        # reduce chunks (if any axis chunksize is > 1)
        if any(s > 1 for i, s in enumerate(result.chunksize) if i in axis):
            args = (result, inds)
            adjust_chunks = {
                i: (1,) * len(c) if i in axis else c
                for i, c in enumerate(result.chunks)
            }
            result = blockwise(
                func,
                inds,
                *args,
                axis=axis,
                keepdims=True,
                dtype=dtype,
                adjust_chunks=adjust_chunks,
            )

    # TODO: [optimization] remove extra squeeze (and materialized zarr) by doing it as a part of the last blockwise
    if not keepdims:
        result = squeeze(result, axis)

    return result


def squeeze(x, /, axis):
    if not isinstance(axis, tuple):
        axis = (axis,)

    if any(x.shape[i] != 1 for i in axis):
        raise ValueError("cannot squeeze axis with size other than one")

    axis = validate_axis(axis, x.ndim)

    chunks = tuple(c for i, c in enumerate(x.chunks) if i not in axis)

    return map_blocks(
        np.squeeze, x, dtype=x.dtype, chunks=chunks, drop_axis=axis, axis=axis
    )
