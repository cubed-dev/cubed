import inspect
import tempfile
import uuid
from datetime import datetime
from functools import lru_cache

import networkx as nx
import zarr

from cubed.backend_array_api import backend_array_to_numpy_array
from cubed.primitive.blockwise import can_fuse_pipelines, fuse
from cubed.runtime.pipeline import visit_nodes
from cubed.runtime.types import CubedPipeline
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
        pipeline=None,
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

        op_name_unique = gensym()

        if pipeline is None:
            # op
            dag.add_node(
                op_name_unique,
                name=op_name_unique,
                op_name=op_name,
                type="op",
                stack_summaries=stack_summaries,
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
                hidden=hidden,
                pipeline=pipeline,
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

    def optimize(self):
        # note there is no need to prune the dag, since the way it is built
        # ensures that only the transitive dependencies of the target arrays are included

        # fuse map blocks
        dag = self.dag.copy()
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

        return Plan(dag)

    def _create_lazy_zarr_arrays(self, dag):
        # find all lazy zarr arrays in dag
        all_pipeline_nodes = []
        lazy_zarr_arrays = []
        reserved_mem_values = []
        for n, d in dag.nodes(data=True):
            if "pipeline" in d and d["pipeline"].reserved_mem is not None:
                reserved_mem_values.append(d["pipeline"].reserved_mem)
                all_pipeline_nodes.append(n)
            if "target" in d and isinstance(d["target"], LazyZarrArray):
                lazy_zarr_arrays.append(d["target"])

        reserved_mem = max(reserved_mem_values, default=0)

        if len(lazy_zarr_arrays) > 0:
            # add new node and edges
            name = "create-arrays"
            op_name = name
            pipeline = create_zarr_arrays(lazy_zarr_arrays, reserved_mem)
            dag.add_node(
                name,
                name=name,
                op_name=op_name,
                type="op",
                pipeline=pipeline,
                projected_mem=pipeline.projected_mem,
                num_tasks=pipeline.num_tasks,
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
    def _finalize_dag(self, optimize_graph: bool = True) -> nx.MultiDiGraph:
        dag = self.optimize().dag if optimize_graph else self.dag.copy()
        dag = self._create_lazy_zarr_arrays(dag)
        return nx.freeze(dag)

    def execute(
        self,
        executor=None,
        callbacks=None,
        optimize_graph=True,
        resume=None,
        spec=None,
        array_names=None,
        **kwargs,
    ):
        dag = self._finalize_dag(optimize_graph=optimize_graph)

        if callbacks is not None:
            [callback.on_compute_start(dag, resume=resume) for callback in callbacks]
        executor.execute_dag(
            dag,
            callbacks=callbacks,
            array_names=array_names,
            resume=resume,
            spec=spec,
            **kwargs,
        )
        if callbacks is not None:
            [callback.on_compute_end(dag) for callback in callbacks]

    def num_tasks(self, optimize_graph=True, resume=None):
        """Return the number of tasks needed to execute this plan."""
        dag = self._finalize_dag(optimize_graph=optimize_graph)
        tasks = 0
        for _, node in visit_nodes(dag, resume=resume):
            pipeline = node["pipeline"]
            tasks += pipeline.num_tasks
        return tasks

    def num_arrays(self, optimize_graph: bool = True) -> int:
        """Return the number of arrays in this plan."""
        dag = self._finalize_dag(optimize_graph=optimize_graph)
        return sum(d.get("type") == "array" for _, d in dag.nodes(data=True))

    def max_projected_mem(self, optimize_graph=True, resume=None):
        """Return the maximum projected memory across all tasks to execute this plan."""
        dag = self._finalize_dag(optimize_graph=optimize_graph)
        projected_mem_values = [
            node["pipeline"].projected_mem
            for _, node in visit_nodes(dag, resume=resume)
        ]
        return max(projected_mem_values) if len(projected_mem_values) > 0 else 0

    def visualize(
        self, filename="cubed", format=None, rankdir="TB", optimize_graph=True
    ):
        dag = self._finalize_dag(optimize_graph=optimize_graph)
        dag = dag.copy()  # make a copy since we mutate the DAG below

        # remove edges from create-arrays output node to avoid cluttering the diagram
        dag.remove_edges_from(list(dag.out_edges("arrays")))

        # remove hidden nodes
        dag.remove_nodes_from(
            list(n for n, d in dag.nodes(data=True) if d.get("hidden", False))
        )

        dag.graph["graph"] = {
            "rankdir": rankdir,
            "label": (
                f"num tasks: {self.num_tasks(optimize_graph=optimize_graph)}\n"
                f"max projected memory: {memory_repr(self.max_projected_mem(optimize_graph=optimize_graph))}"
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
            tooltip = f"name: {n}\n"
            node_type = d.get("type", None)
            if node_type == "op":
                op_name = d["op_name"]
                if op_name == "blockwise":
                    d["style"] = '"rounded,filled"'
                    d["fillcolor"] = "#dcbeff"
                    op_name_summary = "(bw)"
                elif op_name == "rechunk":
                    d["style"] = '"rounded,filled"'
                    d["fillcolor"] = "#aaffc3"
                    op_name_summary = "(rc)"
                else:
                    # creation function
                    d["style"] = "rounded"
                    op_name_summary = ""
                tooltip += f"op: {op_name}"

                if "pipeline" in d:
                    pipeline = d["pipeline"]
                    tooltip += (
                        f"\nprojected memory: {memory_repr(pipeline.projected_mem)}"
                    )
                    tooltip += f"\ntasks: {pipeline.num_tasks}"
                    if pipeline.write_chunks is not None:
                        tooltip += f"\nwrite chunks: {pipeline.write_chunks}"

                    # remove pipeline attribute since it is a long string that causes graphviz to fail
                    del d["pipeline"]

                if "stack_summaries" in d and d["stack_summaries"] is not None:
                    # add call stack information
                    stack_summaries = d["stack_summaries"]

                    first_cubed_i = min(
                        i for i, s in enumerate(stack_summaries) if s.is_cubed()
                    )
                    first_cubed_summary = stack_summaries[first_cubed_i]
                    caller_summary = stack_summaries[first_cubed_i - 1]

                    d["label"] = f"{first_cubed_summary.name} {op_name_summary}"

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

            elif node_type == "array":
                target = d["target"]
                chunkmem = memory_repr(chunk_memory(target.dtype, target.chunks))

                # materialized arrays are light orange, virtual arrays are white
                if isinstance(target, (LazyZarrArray, zarr.Array)):
                    d["style"] = "filled"
                    d["fillcolor"] = "#ffd8b1"
                if n in array_display_names:
                    var_name = array_display_names[n]
                    d["label"] = f"{n} ({var_name})"
                    tooltip += f"variable: {var_name}\n"
                else:
                    d["label"] = n
                tooltip += f"shape: {target.shape}\n"
                tooltip += f"chunks: {target.chunks}\n"
                tooltip += f"dtype: {target.dtype}\n"
                tooltip += f"chunk memory: {chunkmem}\n"

                del d["target"]

            d["tooltip"] = tooltip.strip()

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


def create_zarr_arrays(lazy_zarr_arrays, reserved_mem):
    # projected memory is size of largest initial values, or dtype size if there aren't any
    projected_mem = (
        max(
            [
                # TODO: calculate nbytes from size and dtype itemsize
                backend_array_to_numpy_array(lza.initial_values).nbytes
                if lza.initial_values is not None
                else lza.dtype.itemsize
                for lza in lazy_zarr_arrays
            ],
            default=0,
        )
        + reserved_mem
    )
    num_tasks = len(lazy_zarr_arrays)

    return CubedPipeline(
        create_zarr_array,
        "create_zarr_array",
        lazy_zarr_arrays,
        None,
        None,
        projected_mem,
        reserved_mem,
        num_tasks,
        None,
    )
