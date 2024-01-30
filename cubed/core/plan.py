import inspect
import tempfile
import uuid
from datetime import datetime
from functools import lru_cache
from typing import Callable, Optional

import networkx as nx
import zarr

from cubed.core.optimization import multiple_inputs_optimize_dag
from cubed.primitive.blockwise import BlockwiseSpec
from cubed.primitive.types import PrimitiveOperation
from cubed.runtime.pipeline import visit_nodes
from cubed.runtime.types import ComputeEndEvent, ComputeStartEvent, CubedPipeline
from cubed.storage.zarr import LazyZarrArray
from cubed.utils import chunk_memory, extract_stack_summaries, join_path, memory_repr

# A unique ID with sensible ordering, used for making directory names
CONTEXT_ID = f"cubed-{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4()}"

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

    def __init__(self, dag):
        self.dag = dag

    # args from pipeline onwards are omitted for creation functions when no computation is needed
    @classmethod
    def _new(
        cls,
        name,
        op_name,
        target,
        primitive_op=None,
        hidden=False,
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

        op_name_unique = gensym()

        if primitive_op is None:
            # op
            dag.add_node(
                op_name_unique,
                name=op_name_unique,
                op_name=op_name,
                type="op",
                stack_summaries=stack_summaries,
                op_display_name=f"{op_name_unique}\n{first_cubed_summary.name}",
                hidden=hidden,
            )
            # array (when multiple outputs are supported there could be more than one)
            dag.add_node(
                name,
                name=name,
                type="array",
                target=target,
                hidden=hidden,
            )
            dag.add_edge(op_name_unique, name)
        else:
            # op
            dag.add_node(
                op_name_unique,
                name=op_name_unique,
                op_name=op_name,
                type="op",
                stack_summaries=stack_summaries,
                op_display_name=f"{op_name_unique}\n{first_cubed_summary.name}",
                hidden=hidden,
                primitive_op=primitive_op,
                pipeline=primitive_op.pipeline,
            )
            # array (when multiple outputs are supported there could be more than one)
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

        return Plan(dag)

    @classmethod
    def arrays_to_plan(cls, *arrays):
        return Plan(arrays_to_dag(*arrays))

    def optimize(
        self,
        optimize_function: Optional[Callable[..., nx.MultiDiGraph]] = None,
    ):
        if optimize_function is None:
            optimize_function = multiple_inputs_optimize_dag
        dag = optimize_function(self.dag)
        return Plan(dag)

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
                op_display_name=name,
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

    @lru_cache
    def _finalize_dag(
        self, optimize_graph: bool = True, optimize_function=None
    ) -> nx.MultiDiGraph:
        dag = self.optimize(optimize_function).dag if optimize_graph else self.dag
        # create a copy since _create_lazy_zarr_arrays mutates the dag
        dag = dag.copy()
        dag = self._create_lazy_zarr_arrays(dag)
        return nx.freeze(dag)

    def execute(
        self,
        executor=None,
        callbacks=None,
        optimize_graph=True,
        optimize_function=None,
        resume=None,
        spec=None,
        **kwargs,
    ):
        dag = self._finalize_dag(optimize_graph, optimize_function)

        compute_id = f"compute-{datetime.now().strftime('%Y%m%dT%H%M%S.%f')}"

        if callbacks is not None:
            event = ComputeStartEvent(compute_id, dag, resume)
            [callback.on_compute_start(event) for callback in callbacks]
        executor.execute_dag(
            dag,
            compute_id=compute_id,
            callbacks=callbacks,
            resume=resume,
            spec=spec,
            **kwargs,
        )
        if callbacks is not None:
            event = ComputeEndEvent(compute_id, dag)
            [callback.on_compute_end(event) for callback in callbacks]

    def num_tasks(self, optimize_graph=True, optimize_function=None, resume=None):
        """Return the number of tasks needed to execute this plan."""
        dag = self._finalize_dag(optimize_graph, optimize_function)
        tasks = 0
        for _, node in visit_nodes(dag, resume=resume):
            tasks += node["primitive_op"].num_tasks
        return tasks

    def num_arrays(self, optimize_graph: bool = True, optimize_function=None) -> int:
        """Return the number of arrays in this plan."""
        dag = self._finalize_dag(optimize_graph, optimize_function)
        return sum(d.get("type") == "array" for _, d in dag.nodes(data=True))

    def max_projected_mem(
        self, optimize_graph=True, optimize_function=None, resume=None
    ):
        """Return the maximum projected memory across all tasks to execute this plan."""
        dag = self._finalize_dag(optimize_graph, optimize_function)
        projected_mem_values = [
            node["primitive_op"].projected_mem
            for _, node in visit_nodes(dag, resume=resume)
        ]
        return max(projected_mem_values) if len(projected_mem_values) > 0 else 0

    def total_nbytes_written(
        self, optimize_graph: bool = True, optimize_function=None
    ) -> int:
        """Return the total number of bytes written for all materialized arrays in this plan."""
        dag = self._finalize_dag(optimize_graph, optimize_function)
        nbytes = 0
        for _, d in dag.nodes(data=True):
            if d.get("type") == "array":
                target = d["target"]
                if isinstance(target, LazyZarrArray):
                    nbytes += target.nbytes
        return nbytes

    def visualize(
        self,
        filename="cubed",
        format=None,
        rankdir="TB",
        optimize_graph=True,
        optimize_function=None,
        show_hidden=False,
    ):
        dag = self._finalize_dag(optimize_graph, optimize_function)
        dag = dag.copy()  # make a copy since we mutate the DAG below

        # remove edges from create-arrays output node to avoid cluttering the diagram
        dag.remove_edges_from(list(dag.out_edges("arrays")))

        if not show_hidden:
            dag.remove_nodes_from(
                list(n for n, d in dag.nodes(data=True) if d.get("hidden", False))
            )

        dag.graph["graph"] = {
            "rankdir": rankdir,
            "label": (
                # note that \l is used to left-justify each line (see https://www.graphviz.org/docs/attrs/nojustify/)
                rf"num tasks: {self.num_tasks(optimize_graph, optimize_function)}\l"
                rf"max projected memory: {memory_repr(self.max_projected_mem(optimize_graph, optimize_function))}\l"
                rf"total nbytes written: {memory_repr(self.total_nbytes_written(optimize_graph, optimize_function))}\l"
                rf"optimized: {optimize_graph}\l"
            ),
            "labelloc": "bottom",
            "labeljust": "left",
            "fontsize": "10",
        }
        dag.graph["node"] = {"fontname": "helvetica", "shape": "box", "fontsize": "10"}

        # do an initial pass to extract array variable names from stack summaries
        array_display_names = {}
        for n, d in dag.nodes(data=True):
            if "stack_summaries" in d:
                stack_summaries = d["stack_summaries"]
                first_cubed_i = min(
                    i for i, s in enumerate(stack_summaries) if s.is_cubed()
                )
                caller_summary = stack_summaries[first_cubed_i - 1]
                array_display_names.update(caller_summary.array_names_to_variable_names)
        # add current stack info
        frame = inspect.currentframe().f_back  # go back one in the stack
        stack_summaries = extract_stack_summaries(frame, limit=10)
        first_cubed_i = min(i for i, s in enumerate(stack_summaries) if s.is_cubed())
        caller_summary = stack_summaries[first_cubed_i - 1]
        array_display_names.update(caller_summary.array_names_to_variable_names)

        # now set node attributes with visualization info
        for n, d in dag.nodes(data=True):
            label = n
            tooltip = f"name: {n}\n"
            node_type = d.get("type", None)
            if node_type == "op":
                label = d["op_display_name"]
                op_name = d["op_name"]
                if op_name == "blockwise":
                    d["style"] = '"rounded,filled"'
                    d["fillcolor"] = "#dcbeff"
                elif op_name == "rechunk":
                    d["style"] = '"rounded,filled"'
                    d["fillcolor"] = "#aaffc3"
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
                nbytes = None

                # materialized arrays are light orange, virtual arrays are white
                if isinstance(target, (LazyZarrArray, zarr.Array)):
                    d["style"] = "filled"
                    d["fillcolor"] = "#ffd8b1"
                    nbytes = memory_repr(target.nbytes)
                if n in array_display_names:
                    var_name = array_display_names[n]
                    label = f"{n}\n{var_name}"
                    tooltip += f"variable: {var_name}\n"
                tooltip += f"shape: {target.shape}\n"
                tooltip += f"chunks: {target.chunks}\n"
                tooltip += f"dtype: {target.dtype}\n"
                tooltip += f"chunk memory: {chunkmem}\n"
                if nbytes is not None:
                    tooltip += f"nbytes: {nbytes}\n"

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
                return display.SVG(filename=full_filename)
        except ImportError:
            # Can't return a display object if no IPython.
            pass
        return None


def arrays_to_dag(*arrays):
    from .array import check_array_specs

    check_array_specs(arrays)
    dags = [x.plan.dag for x in arrays if hasattr(x, "plan")]
    return nx.compose_all(dags)


def arrays_to_plan(*arrays):
    plans = [x.plan for x in arrays if hasattr(x, "plan")]
    if len(plans) == 0:
        raise ValueError(f"No plans found for arrays: {arrays}")
    return plans[0].arrays_to_plan(*arrays)


def new_temp_path(name, suffix=".zarr", spec=None):
    work_dir = spec.work_dir if spec is not None else None
    if work_dir is None:
        work_dir = tempfile.gettempdir()
    context_dir = join_path(work_dir, CONTEXT_ID)
    return join_path(context_dir, f"{name}{suffix}")


def create_zarr_array(lazy_zarr_array, *, config=None):
    """Stage function for create."""
    lazy_zarr_array.create(mode="a")


def create_zarr_arrays(lazy_zarr_arrays, allowed_mem, reserved_mem):
    # projected memory is size of largest dtype size (for a fill value)
    projected_mem = (
        max([lza.dtype.itemsize for lza in lazy_zarr_arrays], default=0) + reserved_mem
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
        fusable=False,
    )
