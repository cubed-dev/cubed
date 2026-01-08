import atexit
import dataclasses
import inspect
import shutil
import tempfile
import uuid
import warnings
from datetime import datetime
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import networkx as nx

from cubed.core.optimization import is_input_array, multiple_inputs_optimize_dag
from cubed.primitive.blockwise import BlockwiseSpec
from cubed.primitive.types import PrimitiveOperation
from cubed.runtime.pipeline import visit_node_generations
from cubed.runtime.types import ComputeEndEvent, ComputeStartEvent, CubedPipeline
from cubed.storage.store import is_storage_array
from cubed.storage.zarr import LazyZarrArray, open_if_lazy_zarr_array
from cubed.utils import (
    chunk_memory,
    extract_array_names_from_stack_summaries,
    extract_stack_summaries,
    is_local_path,
    itemsize,
    join_path,
    memory_repr,
)

try:
    from zarr.errors import ArrayNotFoundError  # type: ignore
except ImportError:
    ArrayNotFoundError = FileNotFoundError  # type: ignore # zarr-python<=3.1.1

# A unique ID with sensible ordering, used for making directory names
CONTEXT_ID = f"cubed-{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4()}"

# Delete local context dirs when Python exits
CONTEXT_DIRS = set()

Decorator = Callable


def delete_on_exit(context_dir: str) -> None:
    if context_dir not in CONTEXT_DIRS and is_local_path(context_dir):
        atexit.register(lambda: shutil.rmtree(context_dir, ignore_errors=True))
        CONTEXT_DIRS.add(context_dir)


sym_counter = 0


def gensym(name="op"):
    global sym_counter
    sym_counter += 1
    return f"{name}-{sym_counter:03}"


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

    Compared to a more traditional DAG representing a computation, in Cubed
    nodes are not values that are passed to functions, they are instead
    "parallel computations" which are run for their side effects. Data does
    not flow through the graph - it is written to external storage (Zarr files)
    as the output of one pipeline, then read back as the input to later pipelines.
    """

    def __init__(self, dag, array_names):
        self.dag = dag
        self.array_names = array_names

    # args from primitive_op onwards are omitted for creation functions when no computation is needed
    @classmethod
    def _new(
        cls,
        name,
        op_name,
        target,
        primitive_op=None,
        hidden=False,
        scalar_value=None,
        *source_arrays,
    ):
        # create an empty DAG or combine from sources
        if len(source_arrays) == 0:
            dag = nx.MultiDiGraph()
        else:
            dag = arrays_to_dag(*source_arrays)

        # add new node and edges
        frame = inspect.currentframe().f_back  # go back one in the stack
        stack_summaries = extract_stack_summaries(frame, limit=10)

        first_cubed_i = min(i for i, s in enumerate(stack_summaries) if s.is_cubed())
        first_cubed_summary = stack_summaries[first_cubed_i]
        func_name = first_cubed_summary.name

        op_name_unique = gensym()

        if primitive_op is None:
            # op
            dag.add_node(
                op_name_unique,
                name=op_name_unique,
                op_name=op_name,
                func_name=func_name,
                type="op",
                stack_summaries=stack_summaries,
                hidden=hidden,
            )
            # array
            if isinstance(name, list):  # multiple outputs
                for n, t in zip(name, target):
                    dag.add_node(
                        n,
                        name=n,
                        type="array",
                        target=t,
                        hidden=hidden,
                    )
                    dag.add_edge(op_name_unique, n)
            else:  # single output
                dag.add_node(
                    name,
                    name=name,
                    type="array",
                    target=target,
                    hidden=hidden,
                    scalar_value=scalar_value,
                )
                dag.add_edge(op_name_unique, name)
        else:
            # op
            dag.add_node(
                op_name_unique,
                name=op_name_unique,
                op_name=op_name,
                func_name=func_name,
                type="op",
                stack_summaries=stack_summaries,
                hidden=hidden,
                primitive_op=primitive_op,
                pipeline=primitive_op.pipeline,
            )
            # array
            if isinstance(name, list):  # multiple outputs
                for n, t in zip(name, target):
                    dag.add_node(
                        n,
                        name=n,
                        type="array",
                        target=t,
                        hidden=hidden,
                    )
                    dag.add_edge(op_name_unique, n)
            else:  # single output
                dag.add_node(
                    name,
                    name=name,
                    type="array",
                    target=target,
                    hidden=hidden,
                )
                dag.add_edge(op_name_unique, name)
        for x in source_arrays:
            if hasattr(x, "name"):
                dag.add_edge(x.name, op_name_unique)

        return Plan(dag, (name,))

    @classmethod
    def arrays_to_plan(cls, *arrays):
        return Plan(arrays_to_dag(*arrays), tuple(a.name for a in arrays))

    def optimize(
        self,
        optimize_function: Optional[Callable[..., nx.MultiDiGraph]] = None,
    ):
        if optimize_function is None:
            optimize_function = multiple_inputs_optimize_dag
        dag = optimize_function(self.dag, array_names=self.array_names)
        return Plan(dag, self.array_names)

    def _create_lazy_zarr_arrays(self, dag):
        # find all lazy zarr arrays in dag
        all_pipeline_nodes = []
        lazy_zarr_arrays = []
        allowed_mem = 0
        reserved_mem = 0
        for n, d in dag.nodes(data=True):
            if "primitive_op" in d:
                all_pipeline_nodes.append(n)
                allowed_mem = max(allowed_mem, d["primitive_op"].allowed_mem)
                reserved_mem = max(reserved_mem, d["primitive_op"].reserved_mem)

            if "target" in d and isinstance(d["target"], LazyZarrArray):
                lazy_zarr_arrays.append(d["target"])

        if len(lazy_zarr_arrays) > 0:
            # add new node and edges
            name = "create-arrays"
            op_name = name
            primitive_op = create_zarr_arrays(
                lazy_zarr_arrays, allowed_mem, reserved_mem
            )
            dag.add_node(
                name,
                name=name,
                op_name=op_name,
                type="op",
                func_name="",
                primitive_op=primitive_op,
                pipeline=primitive_op.pipeline,
            )
            dag.add_node(
                "arrays",
                name="arrays",
                target=None,
            )
            dag.add_edge(name, "arrays")
            # make create arrays node a predecessor of all pipeline nodes so it runs first
            for n in all_pipeline_nodes:
                dag.add_edge("arrays", n)

        return dag

    def _compile_blockwise(self, dag, compile_function: Decorator) -> nx.MultiDiGraph:
        """Compiles functions from all blockwise ops by mutating the input dag."""
        # Recommended: make a copy of the dag before calling this function.

        compile_with_config = (
            "config" in inspect.getfullargspec(compile_function).kwonlyargs
        )

        for n in dag.nodes:
            node = dag.nodes[n]

            if "primitive_op" not in node:
                continue

            if not isinstance(node["pipeline"].config, BlockwiseSpec):
                continue

            if compile_with_config:
                compiled = compile_function(
                    node["pipeline"].config.function, config=node["pipeline"].config
                )
            else:
                compiled = compile_function(node["pipeline"].config.function)

            # node is a blockwise primitive_op.
            # maybe we should investigate some sort of optics library for frozen dataclasses...
            new_pipeline = dataclasses.replace(
                node["pipeline"],
                config=dataclasses.replace(node["pipeline"].config, function=compiled),
            )
            node["pipeline"] = new_pipeline

        return dag

    def _find_ops_exceeding_memory(self, dag) -> List[Tuple[str, "PrimitiveOperation"]]:
        """Find all operations where projected memory exceeds allowed memory.

        Returns a list of (op_name, primitive_op) tuples for operations that
        exceed memory limits, sorted by projected memory (highest first).
        """
        ops_exceeding = []
        for n, d in dag.nodes(data=True):
            if "primitive_op" in d:
                op = d["primitive_op"]
                if op.projected_mem > op.allowed_mem:
                    ops_exceeding.append((n, op))
        # Sort by projected_mem descending so worst offenders are first
        ops_exceeding.sort(key=lambda x: x[1].projected_mem, reverse=True)
        return ops_exceeding

    @lru_cache  # noqa: B019
    def _finalize(
        self,
        optimize_graph: bool = True,
        optimize_function=None,
        compile_function: Optional[Decorator] = None,
    ) -> "FinalizedPlan":
        dag = self.optimize(optimize_function).dag if optimize_graph else self.dag
        # create a copy since _create_lazy_zarr_arrays mutates the dag
        dag = dag.copy()
        if callable(compile_function):
            dag = self._compile_blockwise(dag, compile_function)
        dag = self._create_lazy_zarr_arrays(dag)
        ops_exceeding_memory = self._find_ops_exceeding_memory(dag)
        return FinalizedPlan(
            nx.freeze(dag), self.array_names, optimize_graph, ops_exceeding_memory
        )


class ArrayRole(Enum):
    INPUT = "input"
    INTERMEDIATE = "intermediate"
    OUTPUT = "output"


class FinalizedPlan:
    """A plan that is ready to be run.

    Finalizing a plan involves the following steps:
    1. optimization (optional)
    2. adding housekeeping nodes to create arrays
    3. compiling functions (optional)
    4. freezing the final DAG so it can't be changed
    """

    def __init__(self, dag, array_names, optimized, ops_exceeding_memory=None):
        self.dag = dag
        self.array_names = array_names
        self.optimized = optimized
        self._ops_exceeding_memory = ops_exceeding_memory or []
        self._calculate_stats()

        self.input_array_names = []
        self.intermediate_array_names = []
        for name, node in self.dag.nodes(data=True):
            if node.get("type", None) == "array":
                if is_input_array(self.dag, name):
                    self.input_array_names.append(name)
                elif name not in self.array_names:
                    self.intermediate_array_names.append(name)

    def _calculate_stats(self) -> None:
        self._num_stages = len(list(visit_node_generations(self.dag)))

        self._allowed_mem = 0
        self._max_projected_mem = 0
        self._num_arrays = 0
        self._num_primitive_ops = 0
        self._num_tasks = 0
        self._total_narrays_read = 0
        self._total_nbytes_read = 0
        self._total_nchunks_read = 0
        self._total_narrays_written = 0
        self._total_nbytes_written = 0
        self._total_nchunks_written = 0
        self._total_input_narrays = 0
        self._total_input_nbytes = 0
        self._total_input_nchunks = 0
        self._total_intermediate_narrays = 0
        self._total_intermediate_nbytes = 0
        self._total_intermediate_nchunks = 0
        self._total_output_narrays = 0
        self._total_output_nbytes = 0
        self._total_output_nchunks = 0
        self._total_narrays = 0
        self._total_nbytes = 0
        self._total_nchunks = 0

        for name, node in self.dag.nodes(data=True):
            node_type = node.get("type", None)
            if node_type == "op":
                primitive_op = node.get("primitive_op", None)
                if primitive_op is not None:
                    # allowed mem is the same for all ops
                    self._allowed_mem = primitive_op.allowed_mem
                    self._max_projected_mem = max(
                        primitive_op.projected_mem, self._max_projected_mem
                    )
                    self._num_primitive_ops += 1
                    self._num_tasks += primitive_op.num_tasks
            elif node_type == "array":
                array = node["target"]
                self._num_arrays += 1
                if not (isinstance(array, LazyZarrArray) or is_storage_array(array)):
                    continue  # not materialized
                self._total_narrays += 1
                self._total_nbytes += array.nbytes
                self._total_nchunks += array.nchunks
                if is_input_array(self.dag, name):
                    self._total_narrays_read += 1
                    self._total_nbytes_read += array.nbytes
                    self._total_nchunks_read += array.nchunks
                    self._total_input_narrays += 1
                    self._total_input_nbytes += array.nbytes
                    self._total_input_nchunks += array.nchunks
                elif isinstance(array, LazyZarrArray) or is_storage_array(array):
                    self._total_narrays_written += 1
                    self._total_nbytes_written += array.nbytes
                    self._total_nchunks_written += array.nchunks
                    if name in self.array_names:
                        self._total_output_narrays += 1
                        self._total_output_nbytes += array.nbytes
                        self._total_output_nchunks += array.nchunks
                    else:
                        self._total_narrays_read += 1
                        self._total_nbytes_read += array.nbytes
                        self._total_nchunks_read += array.nchunks
                        self._total_intermediate_narrays += 1
                        self._total_intermediate_nbytes += array.nbytes
                        self._total_intermediate_nchunks += array.nchunks

    def array_role(self, name) -> ArrayRole:
        """The role of the array in the computation: an input, intermediate or output."""
        if name in self.input_array_names:
            return ArrayRole.INPUT
        elif name in self.intermediate_array_names:
            return ArrayRole.INTERMEDIATE
        elif name in self.array_names:
            return ArrayRole.OUTPUT
        else:
            raise ValueError(f"Plan does not contain an array with name {name}")

    @property
    def allowed_mem(self) -> int:
        """The total memory available to a worker for running a task, in bytes."""
        return self._allowed_mem

    @property
    def max_projected_mem(self) -> int:
        """The maximum projected memory across all tasks to execute this plan."""
        return self._max_projected_mem

    @property
    def num_arrays(self) -> int:
        """The number of arrays in this plan."""
        return self._num_arrays

    @property
    def num_stages(self) -> int:
        """The number of stages in this plan (including the initial stage to create arrays)."""
        return self._num_stages

    @property
    def num_primitive_ops(self) -> int:
        """The number of primitive operations in this plan."""
        return self._num_primitive_ops

    @property
    def num_tasks(self) -> int:
        """The number of tasks needed to execute this plan."""
        return self._num_tasks

    @property
    def total_narrays_read(self) -> int:
        """The total number of arrays read from for all materialized arrays in this plan."""
        return self._total_narrays_read

    @property
    def total_nbytes_read(self) -> int:
        """The total number of bytes read for all materialized arrays in this plan."""
        return self._total_nbytes_read

    @property
    def total_nchunks_read(self) -> int:
        """The total number of chunks read for all materialized arrays in this plan."""
        return self._total_nchunks_read

    @property
    def total_narrays_written(self) -> int:
        """The total number of arrays written to for all materialized arrays in this plan."""
        return self._total_narrays_written

    @property
    def total_nbytes_written(self) -> int:
        """The total number of bytes written for all materialized arrays in this plan."""
        return self._total_nbytes_written

    @property
    def total_nchunks_written(self) -> int:
        """The total number of chunks written for all materialized arrays in this plan."""
        return self._total_nchunks_written

    @property
    def total_input_narrays(self) -> int:
        """The total number of materialized input arrays in this plan."""
        return self._total_input_narrays

    @property
    def total_input_nbytes(self) -> int:
        """The total number of bytes for all materialized input arrays in this plan."""
        return self._total_input_nbytes

    @property
    def total_input_nchunks(self) -> int:
        """The total number of chunks for all materialized input arrays in this plan."""
        return self._total_input_nchunks

    @property
    def total_intermediate_narrays(self) -> int:
        """The total number of intermediate arrays in this plan."""
        return self._total_intermediate_narrays

    @property
    def total_intermediate_nbytes(self) -> int:
        """The total number of bytes for all intermediate arrays in this plan."""
        return self._total_intermediate_nbytes

    @property
    def total_intermediate_nchunks(self) -> int:
        """The total number of chunks for all intermediate arrays in this plan."""
        return self._total_intermediate_nchunks

    @property
    def total_output_narrays(self) -> int:
        """The total number of output arrays in this plan."""
        return self._total_output_narrays

    @property
    def total_output_nbytes(self) -> int:
        """The total number of bytes for all output arrays in this plan."""
        return self._total_output_nbytes

    @property
    def total_output_nchunks(self) -> int:
        """The total number of chunks for all output arrays in this plan."""
        return self._total_output_nchunks

    @property
    def total_narrays(self) -> int:
        """The total number of materialized arrays in this plan."""
        return self._total_narrays

    @property
    def total_nbytes(self) -> int:
        """The total number of bytes for all materialized arrays in this plan."""
        return self._total_nbytes

    @property
    def total_nchunks(self) -> int:
        """The total number of chunks for all materialized arrays in this plan."""
        return self._total_nchunks

    @property
    def exceeds_memory(self) -> bool:
        """True if any operation in this plan exceeds the allowed memory."""
        return len(self._ops_exceeding_memory) > 0

    @property
    def ops_exceeding_memory(self) -> List[Tuple[str, "PrimitiveOperation"]]:
        """List of (op_name, primitive_op) tuples for operations exceeding memory.

        Sorted by projected memory (highest first).
        """
        return self._ops_exceeding_memory

    def validate(self) -> None:
        """Validate that this plan can be executed.

        Raises
        ------
        ValueError
            If any operation's projected memory exceeds the allowed memory.
        """
        if self._ops_exceeding_memory:
            op_name, op = self._ops_exceeding_memory[0]  # Report worst offender
            raise ValueError(
                f"Projected blockwise memory ({memory_repr(op.projected_mem)}) exceeds allowed_mem ({memory_repr(op.allowed_mem)}), "
                f"including reserved_mem ({memory_repr(op.reserved_mem)}) for {op_name}"
            )

    def execute(
        self,
        executor=None,
        callbacks=None,
        resume=None,
        spec=None,
        **kwargs,
    ):
        self.validate()

        dag = self.dag

        if resume:
            # mark nodes as computed so they are not visited by visit_nodes
            dag = dag.copy()
            nodes = {n: d for (n, d) in dag.nodes(data=True)}
            for name in list(nx.topological_sort(dag)):
                dag.nodes[name]["computed"] = already_computed(name, dag, nodes)

        compute_id = f"compute-{datetime.now().strftime('%Y%m%dT%H%M%S.%f')}"

        if callbacks is not None:
            event = ComputeStartEventWithPlan(compute_id, dag, self)
            [callback.on_compute_start(event) for callback in callbacks]
        executor.execute_dag(
            dag,
            compute_id=compute_id,
            callbacks=callbacks,
            spec=spec,
            **kwargs,
        )
        if callbacks is not None:
            event = ComputeEndEvent(compute_id, dag)
            [callback.on_compute_end(event) for callback in callbacks]

    def visualize(
        self,
        filename="cubed",
        format=None,
        rankdir="TB",
        show_hidden=False,
        engine: Literal["cytoscape", "graphviz"] | None = None,
    ):
        from cubed.diagnostics.colors import APRICOT, LAVENDER, RED

        if engine == "cytoscape":
            return self.visualize_cytoscape(
                filename,
                format=format,
                rankdir=rankdir,
                show_hidden=show_hidden,
            )

        if self._ops_exceeding_memory:
            op_names = [name for name, _ in self._ops_exceeding_memory]
            warnings.warn(
                f"Plan has {len(self._ops_exceeding_memory)} operation(s) that exceed allowed memory: {op_names}. "
                "These are shown in red in the visualization.",
                stacklevel=2,
            )
        ops_exceeding_names = {name for name, _ in self._ops_exceeding_memory}

        dag = self.dag.copy()  # make a copy since we mutate the DAG below

        # remove edges from create-arrays output node to avoid cluttering the diagram
        dag.remove_edges_from(list(dag.out_edges("arrays")))

        if not show_hidden:
            dag.remove_nodes_from(
                list(n for n, d in dag.nodes(data=True) if d.get("hidden", False))
            )

        # Build the graph label - use HTML-like label for mixed colors if memory exceeded
        stats_text = (
            f"num tasks: {self.num_tasks}<BR ALIGN='LEFT'/>"
            f"max projected memory: {memory_repr(self.max_projected_mem)}<BR ALIGN='LEFT'/>"
            f"total nbytes written: {memory_repr(self.total_nbytes_written)}<BR ALIGN='LEFT'/>"
            f"optimized: {self.optimized}<BR ALIGN='LEFT'/>"
        )

        if self._ops_exceeding_memory:
            # Build warning text in red
            warning_lines = [
                "<BR ALIGN='LEFT'/>!!! MEMORY EXCEEDED !!!<BR ALIGN='LEFT'/>"
            ]
            for op_name, op in self._ops_exceeding_memory:
                warning_lines.append(
                    f"{op_name}: requires {memory_repr(op.projected_mem)}, "
                    f"allowed {memory_repr(op.allowed_mem)}<BR ALIGN='LEFT'/>"
                )
            warning_text = "".join(warning_lines)
            # HTML-like label with mixed colors
            label = (
                f"<<FONT>{stats_text}</FONT><FONT COLOR='{RED}'>{warning_text}</FONT>>"
            )
        else:
            # Simple HTML label (no warning)
            label = f"<{stats_text}>"

        dag.graph["graph"] = {
            "rankdir": rankdir,
            "label": label,
            "labelloc": "bottom",
            "labeljust": "left",
            "fontsize": "10",
        }

        dag.graph["node"] = {"fontname": "helvetica", "shape": "box", "fontsize": "10"}

        # do an initial pass to extract array variable names from stack summaries
        stacks = []
        for _, d in dag.nodes(data=True):
            if "stack_summaries" in d:
                stack_summaries = d["stack_summaries"]
                stacks.append(stack_summaries)
        # add current stack info
        # go back one in the stack to the caller of 'visualize'
        frame = inspect.currentframe()
        frame = frame.f_back if frame is not None else frame
        stack_summaries = extract_stack_summaries(frame, limit=10)
        stacks.append(stack_summaries)
        array_display_names = extract_array_names_from_stack_summaries(stacks)

        # now set node attributes with visualization info
        for n, d in dag.nodes(data=True):
            label = n
            tooltip = f"name: {n}\n"
            node_type = d.get("type", None)
            if node_type == "op":
                func_name = d["func_name"]
                label = f"{n}\n{func_name}".strip()
                op_name = d["op_name"]
                if n in ops_exceeding_names:
                    # operation exceeds memory - show in red
                    d["style"] = '"rounded,filled"'
                    d["fillcolor"] = RED
                elif op_name == "blockwise" or op_name == "rechunk":
                    d["style"] = '"rounded,filled"'
                    d["fillcolor"] = LAVENDER
                else:
                    # creation function
                    d["style"] = "rounded"
                tooltip += f"op: {op_name}"

                num_tasks = None
                if "primitive_op" in d:
                    primitive_op = d["primitive_op"]
                    tooltip += (
                        f"\nprojected memory: {memory_repr(primitive_op.projected_mem)}"
                    )
                    num_tasks = primitive_op.num_tasks
                    tooltip += f"\ntasks: {num_tasks}"
                    if primitive_op.write_chunks is not None:
                        tooltip += f"\nwrite chunks: {primitive_op.write_chunks}"
                    del d["primitive_op"]

                # remove pipeline attribute since it is a long string that causes graphviz to fail
                if "pipeline" in d:
                    pipeline = d["pipeline"]
                    if isinstance(pipeline.config, BlockwiseSpec):
                        tooltip += (
                            f"\nnum input blocks: {pipeline.config.num_input_blocks}"
                        )
                        tooltip += (
                            f"\nnum output blocks: {pipeline.config.num_output_blocks}"
                        )
                    del d["pipeline"]

                if "stack_summaries" in d and d["stack_summaries"] is not None:
                    # add call stack information
                    stack_summaries = d["stack_summaries"]

                    first_cubed_i = min(
                        i for i, s in enumerate(stack_summaries) if s.is_cubed()
                    )
                    caller_summary = stack_summaries[first_cubed_i - 1]

                    calls = " -> ".join(
                        [
                            s.name
                            for s in stack_summaries
                            if not s.is_on_python_lib_path()
                        ]
                    )

                    line = f"{caller_summary.lineno} in {caller_summary.name}"

                    tooltip += f"\ncalls: {calls}"
                    tooltip += f"\nline: {line}"
                    del d["stack_summaries"]

                if num_tasks is not None:
                    label += f"\ntasks: {num_tasks}"

            elif node_type == "array":
                target = d["target"]
                chunkmem = memory_repr(chunk_memory(target))

                # materialized arrays are light orange, virtual arrays are white
                if isinstance(target, LazyZarrArray) or is_storage_array(target):
                    d["style"] = "filled"
                    d["fillcolor"] = APRICOT
                if n in array_display_names:
                    var_name = array_display_names[n]
                    label = f"{n}\n{var_name}"
                    tooltip += f"variable: {var_name}\n"
                if "scalar_value" in d and d["scalar_value"] is not None:
                    scalar_value = d["scalar_value"]
                    label += f"\n{scalar_value}"
                    tooltip += f"value: {scalar_value}\n"
                tooltip += f"shape: {target.shape}\n"
                tooltip += f"chunks: {target.chunks}\n"
                tooltip += f"dtype: {target.dtype}\n"
                tooltip += f"chunk memory: {chunkmem}\n"
                if hasattr(target, "nbytes"):
                    tooltip += f"nbytes: {memory_repr(target.nbytes)}\n"
                if hasattr(target, "nchunks"):
                    tooltip += f"nchunks: {target.nchunks}\n"

                del d["target"]

            # quote strings with colons in them (https://github.com/pydot/pydot/issues/258)
            d["label"] = '"' + label.strip() + '"'
            d["tooltip"] = '"' + tooltip.strip() + '"'

            if "name" in d:  # pydot already has name
                del d["name"]
        gv = nx.drawing.nx_pydot.to_pydot(dag)
        if format is None:
            format = "svg"
        full_filename = f"{filename}.{format}"
        gv.write(full_filename, format=format)

        try:  # pragma: no cover
            import IPython.display as display

            if format == "svg":
                return display.HTML(filename=full_filename)
        except ImportError:
            # Can't return a display object if no IPython.
            pass
        return None

    def visualize_cytoscape(
        self,
        filename="cubed",
        format=None,
        rankdir="TB",
        show_hidden=False,
    ):
        from cubed.diagnostics.widgets.plan import create_or_update_plan_widget

        widget = create_or_update_plan_widget(self, rankdir=rankdir)

        if filename is not None:
            from ipywidgets.embed import embed_minimal_html

            if format is None:
                format = "html"
            full_filename = f"{filename}.{format}"
            embed_minimal_html(
                full_filename, views=[widget], title="Cubed plan", drop_defaults=False
            )
        return widget


@dataclasses.dataclass
class ComputeStartEventWithPlan(ComputeStartEvent):
    plan: FinalizedPlan
    """The plan."""


def arrays_to_dag(*arrays):
    from .array import check_array_specs

    check_array_specs(arrays)
    dags = [x._plan.dag for x in arrays if hasattr(x, "_plan")]
    return nx.compose_all(dags)


def arrays_to_plan(*arrays):
    plans = [x._plan for x in arrays if hasattr(x, "_plan")]
    if len(plans) == 0:
        raise ValueError(f"No plans found for arrays: {arrays}")
    return plans[0].arrays_to_plan(*arrays)


def intermediate_store(spec=None):
    """Return a file path or a store object that is used for storing
    intermediate data.

    By default returns a temporary file path, which may be local or remote.
    """
    if spec.intermediate_store is not None:
        return spec.intermediate_store
    work_dir = spec.work_dir if spec is not None else None
    if work_dir is None:
        work_dir = tempfile.gettempdir()
    context_dir = join_path(work_dir, CONTEXT_ID)
    delete_on_exit(context_dir)
    return context_dir


def create_zarr_array(lazy_zarr_array, *, config=None):
    """Stage function for create."""
    lazy_zarr_array.create(mode="a")


def create_zarr_arrays(lazy_zarr_arrays, allowed_mem, reserved_mem):
    # projected memory is size of largest dtype size (for a fill value)
    projected_mem = (
        max([itemsize(lza.dtype) for lza in lazy_zarr_arrays], default=0) + reserved_mem
    )
    num_tasks = len(lazy_zarr_arrays)

    pipeline = CubedPipeline(
        create_zarr_array,
        "create_zarr_array",
        lazy_zarr_arrays,
        None,
    )
    return PrimitiveOperation(
        pipeline=pipeline,
        source_array_names=[],
        target_array=None,
        projected_mem=projected_mem,
        allowed_mem=allowed_mem,
        reserved_mem=reserved_mem,
        num_tasks=num_tasks,
        fusable_with_predecessors=False,
    )


def already_computed(name, dag, nodes: Dict[str, Any]) -> bool:
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

    for output in dag.successors(name):
        target = nodes[output].get("target", None)
        if target is not None:
            try:
                target = open_if_lazy_zarr_array(target)
                if not hasattr(target, "nchunks_initialized"):
                    raise NotImplementedError(
                        f"Zarr array type {type(target)} does not support resume since it doesn't have a 'nchunks_initialized' property"
                    )
                # this check can be expensive since it has to list the directory to find nchunks_initialized
                if target.ndim == 0 or target.nchunks_initialized != target.nchunks:
                    return False
            except ArrayNotFoundError:
                return False
    return True
