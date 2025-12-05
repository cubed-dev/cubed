import inspect
import pathlib

import anywidget
import traitlets
from IPython.display import display

from cubed.core.optimization import successors_unordered
from cubed.core.plan import ArrayRole
from cubed.diagnostics.widgets import get_template
from cubed.runtime.pipeline import visit_nodes
from cubed.runtime.types import Callback
from cubed.storage.store import is_storage_array
from cubed.storage.zarr import LazyZarrArray
from cubed.utils import (
    chunk_memory,
    extract_array_names_from_stack_summaries,
    extract_stack_summaries,
    format_int,
    normalize_chunks,
)
from cubed.vendor.dask.array.svg import svg

LINE_COLOR = "black"
VIRTUAL_LINE_COLOR = "#c7c7c7"

INITIALIZED_ARRAY_BACKGROUND_COLOR = "white"
STORED_ARRAY_BACKGROUND_COLOR = "#aaffc3"
VIRTUAL_ARRAY_BACKGROUND_COLOR = "#e0e0e0"

PRIMITIVE_OP_BACKGROUND_COLOR = "#dcbeff"
VIRTUAL_OP_BACKGROUND_COLOR = "#e0e0e0"

HIGHLIGHT_COLOR = "#c9dbfd"


class PlanWidget(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / "static" / "plan_widget.js"
    _css = pathlib.Path(__file__).parent / "static" / "plan_widget.css"

    height = traitlets.Unicode("600px").tag(sync=True)
    selected_node = traitlets.Unicode().tag(sync=True)
    _summary_html = traitlets.Unicode().tag(sync=True)
    _node_html_dict = traitlets.Dict({}).tag(sync=True)
    _cytoscape_elements = traitlets.List([]).tag(sync=True)
    _cytoscape_layout = traitlets.Dict({}).tag(sync=True)
    _cytoscape_style = traitlets.List([]).tag(sync=True)
    _cytoscape_options = traitlets.Dict({}).tag(sync=True)


class LivePlanViewer(Callback):
    def __init__(self, widget=None, *, height="600px"):
        self.widget = widget
        self.height = height

    def on_compute_start(self, event):
        self.op_name_to_array_names = ops_to_arrays(event.dag)
        self.num_tasks = {}
        self.progress = {}
        for name, node in visit_nodes(event.dag):
            self.num_tasks[name] = node["primitive_op"].num_tasks
            self.progress[name] = 0.0

        if self.widget is None:
            self.widget = create_or_update_plan_widget(event.plan, height=self.height)

            # show the widget in the current cell
            display(self.widget)
        else:
            self.widget = create_or_update_plan_widget(event.plan, widget=self.widget)

        self.widget.send({"type": "on_compute_start"})

    def on_operation_start(self, event):
        self.widget.send({"type": "on_operation_start", "op_name": event.name})

    def on_operation_end(self, event):
        self.widget.send({"type": "on_operation_end", "op_name": event.name})

    def on_task_end(self, event):
        self.progress[event.name] += event.num_tasks / self.num_tasks[event.name]
        self.widget.send(
            {
                "type": "on_task_end",
                "op_name": event.name,
                "array_names": self.op_name_to_array_names[event.name],
                "progress": self.progress[event.name],
            }
        )


def create_or_update_plan_widget(plan, widget=None, height="600px", rankdir="TB"):
    array_display_names, elements = plan_to_cytoscape(plan)

    layout = {
        "name": "dagre",
        "rankDir": rankdir,
        "rankSep": 36,
        "nodeDimensionsIncludeLabels": True,
        "fit": False,
    }
    style = [
        {
            "selector": "node",
            "style": {
                "font-family": "helvetica",
                "font-size": "12",
                "color": "black",
                "background-color": "data(fillcolor)",
                "border-color": "data(linecolor)",
                "border-width": 2,
                "border-style": "data(borderstyle)",
                "opacity": "1.0",
                "text-valign": "center",
                "text-halign": "center",
                "label": "data(label)",
                "shape": "data(shape)",
                "text-wrap": "wrap",
                # note following is deprecated, see https://stackoverflow.com/a/78033670
                "width": "label",
                "height": 36,
                "line-height": 1.2,
                "padding": 10,
            },
        },
        {
            "selector": "edge",
            "style": {
                "width": 2,
                "line-color": "black",
                "line-cap": "square",
                "target-arrow-shape": "triangle",
                "target-arrow-color": "black",
                "curve-style": "bezier",
                "source-endpoint": "outside-to-node",
            },
        },
        {
            "selector": "node:selected",
            "style": {"underlay-color": HIGHLIGHT_COLOR, "underlay-opacity": "0.5"},
        },
    ]
    options = {
        "autoungrabify": True,
        "minZoom": 0.1,
        "maxZoom": 3,
    }

    _summary_html = get_template("plan.html.j2").render(plan=plan)

    _node_html_dict = {}
    for n, d in plan.dag.nodes(data=True):
        node_data = dict(d)
        node_type = node_data.get("type", None)
        if node_type == "op":
            if (
                "stack_summaries" in node_data
                and node_data["stack_summaries"] is not None
            ):
                # add call stack information
                stack_summaries = node_data["stack_summaries"]

                first_cubed_i = min(
                    i for i, s in enumerate(stack_summaries) if s.is_cubed()
                )
                caller_summary = stack_summaries[first_cubed_i - 1]

                calls = " -> ".join(
                    [s.name for s in stack_summaries if not s.is_on_python_lib_path()]
                )

                line = f"{caller_summary.lineno} in {caller_summary.name}"

                node_data["calls"] = calls
                node_data["line"] = line

            node_data_html = get_template("plan_op.html.j2").render(node_data=node_data)
        elif node_type == "array":
            if n in array_display_names:
                node_data["variable_name"] = array_display_names[n]
            node_data["chunk_memory"] = chunk_memory(node_data["target"])
            node_data["role"] = plan.array_role(n).value
            node_data["stored"] = hasattr(node_data["target"], "store")
            if hasattr(node_data["target"], "store"):
                node_data["store"] = node_data["target"].store
            try:
                grid = array_to_svg(node_data["target"])
            except NotImplementedError:
                grid = ""
            node_data_html = get_template("plan_array.html.j2").render(
                node_data=node_data,
                grid=grid,
            )
        else:
            node_data_html = ""
        _node_html_dict[n] = node_data_html

    if widget is None:
        widget = PlanWidget(
            height=height,
            _summary_html=_summary_html,
            _node_html_dict=_node_html_dict,
            _cytoscape_elements=elements,
            _cytoscape_layout=layout,
            _cytoscape_style=style,
            _cytoscape_options=options,
        )
    else:
        widget._summary_html = _summary_html
        widget._node_html_dict = _node_html_dict
        widget._cytoscape_elements = elements
        widget._cytoscape_layout = layout
        widget._cytoscape_style = style
        widget._cytoscape_options = options
    return widget


def plan_to_cytoscape(
    plan,
    show_hidden=False,
):
    dag = plan.dag.copy()  # make a copy since we mutate the DAG below

    # remove edges from create-arrays output node to avoid cluttering the diagram
    dag.remove_edges_from(list(dag.out_edges("arrays")))

    if not show_hidden:
        dag.remove_nodes_from(
            list(n for n, d in dag.nodes(data=True) if d.get("hidden", False))
        )

    # do an initial pass to extract array variable names from stack summaries
    stacks = []
    for _, d in dag.nodes(data=True):
        if "stack_summaries" in d:
            stack_summaries = d["stack_summaries"]
            stacks.append(stack_summaries)
    # add current stack info
    # TODO: following isn't right yet
    # go back one in the stack to the caller of 'compute'
    frame = inspect.currentframe().f_back
    stack_summaries = extract_stack_summaries(frame, limit=10)
    stacks.append(stack_summaries)
    array_display_names = extract_array_names_from_stack_summaries(stacks)

    elements = []

    # now set node attributes with visualization info
    for n, d in dag.nodes(data=True):
        label = n
        node_type = d.get("type", None)
        if node_type == "op":
            func_name = d["func_name"]
            label = f"{n}\n{func_name}".strip()
            num_tasks = None
            if "primitive_op" in d:
                primitive_op = d["primitive_op"]
                num_tasks = primitive_op.num_tasks
                linecolor = LINE_COLOR
                fillcolor = PRIMITIVE_OP_BACKGROUND_COLOR
            else:
                linecolor = VIRTUAL_LINE_COLOR
                fillcolor = VIRTUAL_OP_BACKGROUND_COLOR

            if num_tasks is not None:
                label += f"\ntasks: {format_int(num_tasks)}"

            elements.append(
                {
                    "data": {
                        "id": n,
                        "label": label,
                        "shape": "round-rectangle",
                        "fillcolor": fillcolor,
                        "linecolor": linecolor,
                        "borderstyle": "solid",
                    }
                }
            )

        elif node_type == "array":
            target = d["target"]

            if isinstance(target, LazyZarrArray) or is_storage_array(target):
                linecolor = LINE_COLOR
                fillcolor = INITIALIZED_ARRAY_BACKGROUND_COLOR
            else:
                linecolor = VIRTUAL_LINE_COLOR
                fillcolor = VIRTUAL_ARRAY_BACKGROUND_COLOR
            if n in array_display_names:
                var_name = array_display_names[n]
                label = f"{n}\n{var_name}"

            if plan.array_role(n) == ArrayRole.INTERMEDIATE:
                borderstyle = "dashed"
            else:
                borderstyle = "solid"

            elements.append(
                {
                    "data": {
                        "id": n,
                        "label": label,
                        "shape": "rectangle",
                        "fillcolor": fillcolor,
                        "linecolor": linecolor,
                        "borderstyle": borderstyle,
                    }
                }
            )

        else:
            elements.append(
                {
                    "data": {
                        "id": n,
                        "label": label,
                        "shape": "rectangle",
                        "fillcolor": VIRTUAL_ARRAY_BACKGROUND_COLOR,
                        "linecolor": VIRTUAL_LINE_COLOR,
                        "borderstyle": "solid",
                    }
                }
            )

    for source, target in dag.edges():
        elements.append({"data": {"source": source, "target": target}})

    return array_display_names, elements


def ops_to_arrays(dag):
    """Return a map from op name to the names of the arrays it produces"""
    op_name_to_array_names = {}
    for n, d in dag.nodes(data=True):
        node_type = d.get("type", None)
        if node_type == "op" and "primitive_op" in d:
            op_name_to_array_names[n] = list(successors_unordered(dag, n))

    return op_name_to_array_names


def array_to_svg(array):
    chunks = normalize_chunks(array.chunks, shape=array.shape, dtype=array.dtype)
    return svg(chunks, size=250)
