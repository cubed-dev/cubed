import inspect
import tempfile
import uuid
from datetime import datetime

import networkx as nx

from cubed.primitive.blockwise import can_fuse_pipelines, fuse
from cubed.primitive.types import CubedPipeline
from cubed.runtime.pipeline import already_computed
from cubed.storage.zarr import LazyZarrArray
from cubed.utils import chunk_memory, extract_stack_summaries, join_path, memory_repr
from cubed.vendor.rechunker.types import PipelineExecutor, Stage

# A unique ID with sensible ordering, used for making directory names
CONTEXT_ID = f"cubed-{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4()}"


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

    def __init__(self, dag):
        self.dag = dag

    # args from pipeline onwards are omitted for creation functions when no computation is needed
    @classmethod
    def _new(
        cls,
        name,
        op_name,
        target,
        intermediate_target=None,
        pipeline=None,
        projected_mem=None,
        reserved_mem=None,
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
        stack_summaries = extract_stack_summaries(frame, limit=10)

        if pipeline is None:
            dag.add_node(
                name,
                name=name,
                op_name=op_name,
                target=target,
                stack_summaries=stack_summaries,
            )
        else:
            dag.add_node(
                name,
                name=name,
                op_name=op_name,
                target=target,
                stack_summaries=stack_summaries,
                pipeline=pipeline,
                projected_mem=projected_mem,
                reserved_mem=reserved_mem,
                num_tasks=num_tasks,
            )
        if intermediate_target is not None:
            intermediate_name = f"{name}-intermediate"
            dag.add_node(
                intermediate_name,
                name=intermediate_name,
                op_name=op_name,
                target=intermediate_target,
                stack_summaries=stack_summaries,
                hidden=True,
            )
        for x in source_arrays:
            if hasattr(x, "name"):
                dag.add_edge(x.name, name)

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
            # node must have a single predecessor
            #   - not multiple edges pointing to a single predecessor
            # node must be the single successor to the predecessor
            # and both must have pipelines that can be fused
            if dag.in_degree(n) != 1:
                return False
            pre = next(dag.predecessors(n))
            if dag.out_degree(pre) != 1:
                return False
            if "pipeline" not in nodes[pre] or "pipeline" not in nodes[n]:
                return False
            return can_fuse_pipelines(nodes[pre]["pipeline"], nodes[n]["pipeline"])

        for n in list(dag.nodes()):
            if can_fuse(n):
                pre = next(dag.predecessors(n))
                pipeline = fuse(nodes[pre]["pipeline"], nodes[n]["pipeline"])
                nodes[n]["pipeline"] = pipeline
                assert nodes[n]["target"] == pipeline.target_array

                for p in dag.predecessors(pre):
                    dag.add_edge(p, n)
                dag.remove_node(pre)

        return Plan(dag)

    def create_lazy_zarr_arrays(self, dag):
        # find all lazy zarr arrays in dag
        all_array_nodes = []
        lazy_zarr_arrays = []
        reserved_mem_values = []
        for n, d in dag.nodes(data=True):
            if "reserved_mem" in d and d["reserved_mem"] is not None:
                reserved_mem_values.append(d["reserved_mem"])
            if isinstance(d["target"], LazyZarrArray):
                all_array_nodes.append(n)
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
                target=None,
                pipeline=pipeline,
                projected_mem=pipeline.projected_mem,
                reserved_mem=reserved_mem,
                num_tasks=pipeline.num_tasks,
            )
            # make create arrays node a dependency of all lazy array nodes
            for n in all_array_nodes:
                dag.add_edge(name, n)

        return dag

    def execute(
        self,
        executor=None,
        callbacks=None,
        optimize_graph=True,
        resume=None,
        array_names=None,
        **kwargs,
    ):
        dag = self.optimize().dag if optimize_graph else self.dag.copy()
        dag = self.create_lazy_zarr_arrays(dag)

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
                [
                    callback.on_compute_start(dag, resume=resume)
                    for callback in callbacks
                ]
            executor.execute_dag(
                dag,
                callbacks=callbacks,
                array_names=array_names,
                resume=resume,
                **kwargs,
            )
            if callbacks is not None:
                [callback.on_compute_end(dag) for callback in callbacks]

    def num_tasks(self, optimize_graph=True, resume=None):
        """Return the number of tasks needed to execute this plan."""
        dag = self.optimize().dag if optimize_graph else self.dag.copy()
        dag = self.create_lazy_zarr_arrays(dag)
        tasks = 0
        for _, node in visit_nodes(dag, resume=resume):
            pipeline = node["pipeline"]
            tasks += pipeline.num_tasks
        return tasks

    def num_arrays(self, optimize_graph: bool = True) -> int:
        """Return the number of arrays in this plan."""
        dag = self.optimize().dag if optimize_graph else self.dag
        return dag.number_of_nodes()

    def max_projected_mem(self, optimize_graph=True, resume=None):
        """Return the maximum projected memory across all tasks to execute this plan."""
        dag = self.optimize().dag if optimize_graph else self.dag.copy()
        dag = self.create_lazy_zarr_arrays(dag)
        projected_mem_values = [
            node["pipeline"].projected_mem
            for _, node in visit_nodes(dag, resume=resume)
        ]
        return max(projected_mem_values) if len(projected_mem_values) > 0 else 0

    def visualize(
        self, filename="cubed", format=None, rankdir="TB", optimize_graph=True
    ):
        dag = self.optimize().dag if optimize_graph else self.dag.copy()
        dag = self.create_lazy_zarr_arrays(dag)

        # remove edges from create-arrays node to avoid cluttering the diagram
        dag.remove_edges_from(list(dag.out_edges("create-arrays")))

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
            if d["op_name"] == "blockwise":
                d["style"] = "filled"
                d["fillcolor"] = "#dcbeff"
                op_name_summary = "(bw)"
            elif d["op_name"] == "rechunk":
                d["style"] = "filled"
                d["fillcolor"] = "#aaffc3"
                op_name_summary = "(rc)"
            else:  # creation function
                op_name_summary = ""
            target = d["target"]
            if target is not None:
                chunkmem = memory_repr(chunk_memory(target.dtype, target.chunks))
                tooltip = (
                    f"name: {n}\n"
                    f"shape: {target.shape}\n"
                    f"chunks: {target.chunks}\n"
                    f"dtype: {target.dtype}\n"
                    f"chunk memory: {chunkmem}\n"
                )
            else:
                tooltip = ""
            if "pipeline" in d:
                pipeline = d["pipeline"]
                tooltip += f"\nprojected memory: {memory_repr(pipeline.projected_mem)}"
                tooltip += f"\ntasks: {pipeline.num_tasks}"
            if "stack_summaries" in d and d["stack_summaries"] is not None:
                # add call stack information
                stack_summaries = d["stack_summaries"]

                first_cubed_i = min(
                    i for i, s in enumerate(stack_summaries) if s.is_cubed()
                )
                first_cubed_summary = stack_summaries[first_cubed_i]
                caller_summary = stack_summaries[first_cubed_i - 1]

                if n in array_display_names:
                    var_name = f" ({array_display_names[n]})"
                else:
                    var_name = ""
                d[
                    "label"
                ] = f"{n}{var_name}\n{first_cubed_summary.name} {op_name_summary}"

                calls = " -> ".join(
                    [s.name for s in stack_summaries if not s.is_on_python_lib_path()]
                )

                line = f"{caller_summary.lineno} in {caller_summary.name}"

                tooltip += f"\ncalls: {calls}"
                tooltip += f"\nline: {line}"
                del d["stack_summaries"]

            d["tooltip"] = tooltip.strip()

            # remove pipeline attribute since it is a long string that causes graphviz to fail
            if "pipeline" in d:
                del d["pipeline"]
            if "target" in d:
                del d["target"]
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


def create_zarr_array(lazy_zarr_array, *, config=None):
    """Stage function for create."""
    lazy_zarr_array.create(mode="a")


def create_zarr_arrays(lazy_zarr_arrays, reserved_mem):
    stages = [
        Stage(
            create_zarr_array,
            "create_zarr_array",
            mappable=lazy_zarr_arrays,
        )
    ]

    # projected memory is size of largest initial values, or dtype size if there aren't any
    projected_mem = (
        max(
            [
                lza.initial_values.nbytes
                if lza.initial_values is not None
                else lza.dtype.itemsize
                for lza in lazy_zarr_arrays
            ],
            default=0,
        )
        + reserved_mem
    )
    num_tasks = len(lazy_zarr_arrays)

    return CubedPipeline(stages, None, None, None, projected_mem, num_tasks)
