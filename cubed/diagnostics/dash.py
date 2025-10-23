import inspect
import threading
from math import cos, pi
from pathlib import Path
from random import random

import dash_cytoscape
import dash_svg as dsvg
from dash import Dash, Input, Output, callback, dcc, html
from dash.exceptions import PreventUpdate

from cubed.core.optimization import successors_unordered
from cubed.core.plan import ArrayRole
from cubed.primitive.blockwise import BlockwiseSpec
from cubed.runtime.pipeline import visit_nodes
from cubed.runtime.types import Callback
from cubed.storage.store import is_storage_array
from cubed.storage.zarr import LazyZarrArray
from cubed.utils import (
    chunk_memory,
    extract_stack_summaries,
    memory_repr,
    normalize_chunks,
)

LINE_COLOR = "black"
VIRTUAL_LINE_COLOR = "#c7c7c7"

INITIALIZED_ARRAY_BACKGROUND_COLOR = "white"
STORED_ARRAY_BACKGROUND_COLOR = "#aaffc3"
VIRTUAL_ARRAY_BACKGROUND_COLOR = "#e0e0e0"

PRIMITIVE_OP_BACKGROUND_COLOR = "#dcbeff"
VIRTUAL_OP_BACKGROUND_COLOR = "#e0e0e0"

HIGHLIGHT_COLOR = "#c9dbfd"

# Load dagre layout engine
dash_cytoscape.load_extra_layouts()


class Dashboard(Callback):
    def __init__(self, *args, **kwargs):
        self.done = False
        self.dash_args = args
        self.dash_kwargs = kwargs

    def create_dash_app(self, plan):
        assets_folder = str(Path(__file__).parent / "assets")
        app = Dash(title="Cubed Dashboard", assets_folder=assets_folder)
        cyto, layout, array_display_names = plan_to_cytoscape(plan)
        op_name_to_array_names = ops_to_arrays(plan.dag)

        self.default_stylesheet = cyto.stylesheet

        app.layout = html.Div(
            children=[
                html.Div(
                    children=[
                        html.Img(src="/assets/cubed-logo.png"),
                        plan_to_html(plan),
                        html.Div(
                            [
                                html.H3("Details"),
                                html.Div(id="node-details"),
                            ],
                            className="info-panel",
                        ),
                    ],
                    id="info-container",
                ),
                html.Div(children=[cyto], id="graph-container"),
                html.Div(
                    [
                        html.Button("Reset", id="reset"),
                        html.Button("Fit", id="fit-button"),
                    ],
                    id="cytoscape-controls",
                ),
                dcc.Interval(id="interval-component", interval=100, n_intervals=0),
            ],
        )

        @app.callback(
            Output("cytoscape-component", "zoom"),
            Output("cytoscape-component", "pan"),
            Input("reset", "n_clicks"),
            prevent_initial_call=False,
        )
        def reset_graph(n_clicks):
            r = random() / 1000.0
            # add a tiny random offset so value isn't cached
            return 1.0 + r, {"x": 10.0 + r, "y": 10.0 + r}

        @app.callback(
            Output("cytoscape-component", "layout"),
            Input("fit-button", "n_clicks"),
            prevent_initial_call=True,
        )
        def fit_graph(n_clicks):
            if n_clicks:
                # add random attribute so value isn't cached
                return layout | {"fit": True, "_cacheBuster": random()}
            raise PreventUpdate

        @callback(
            Output("cytoscape-component", "stylesheet"),
            Output("interval-component", "disabled"),
            Input("interval-component", "n_intervals"),
        )
        def update_graph_live(n):
            # set op node background fill opacity to progress value
            stylesheet = list(self.default_stylesheet)
            for name in self.running_operations:
                opacity = (cos(2 * pi * n / 10) + 1) / 2
                stylesheet.append(
                    {
                        "selector": f"node[id = '{name}']",
                        "style": {
                            "background-opacity": f"{opacity}",
                        },
                    }
                )
                progress = self.progress[name]
                pct = int(progress * 100)
                for array_name in op_name_to_array_names.get(name, []):
                    stylesheet.append(
                        {
                            "selector": f"node[id = '{array_name}']",
                            "style": {
                                # use linear gradient as progress bar
                                "background-fill": "linear-gradient",
                                "background-gradient-stop-colors": f"{STORED_ARRAY_BACKGROUND_COLOR} {INITIALIZED_ARRAY_BACKGROUND_COLOR}",
                                "background-gradient-stop-positions": f"{pct}% {pct}%",
                                "background-gradient-direction": "to-right",
                            },
                        }
                    )
            for name in self.completed_operations:
                stylesheet.append(
                    {
                        "selector": f"node[id = '{name}']",
                        "style": {
                            "background-opacity": 1.0,
                        },
                    }
                )
                for array_name in op_name_to_array_names.get(name, []):
                    stylesheet.append(
                        {
                            "selector": f"node[id = '{array_name}']",
                            "style": {
                                "background-color": STORED_ARRAY_BACKGROUND_COLOR,
                                "background-opacity": 1.0,
                            },
                        }
                    )
            # return self.done to disable interval updates when done
            return stylesheet, self.done

        @callback(
            Output("node-details", "children"),
            Input("cytoscape-component", "tapNodeData"),
            Input("cytoscape-component", "selectedNodeData"),
        )
        def display_node_details(data, selected_data):
            # TODO: don't allow multiple selections
            if data and selected_data and len(selected_data) > 0:
                for n, d in self.dag.nodes(data=True):
                    if n == data["id"]:
                        node_type = d.get("type", None)
                        if node_type == "op":
                            return op_to_html(d)
                        elif node_type == "array":
                            return array_to_html(d, array_display_names, plan)

            return "Click on a node in the graph"

        return app

    def on_compute_start(self, event):
        self.dag = event.dag
        app = self.create_dash_app(event.plan)
        run_dash_in_background_thread(app, *self.dash_args, **self.dash_kwargs)

        self.num_tasks = {}
        self.completed_tasks = {}
        self.progress = {}
        self.running_operations = set()
        self.completed_operations = set()
        for name, node in visit_nodes(event.dag):
            self.num_tasks[name] = node["primitive_op"].num_tasks
            self.progress[name] = 0.0
            self.completed_tasks[name] = 0

    def on_compute_end(self, event):
        self.done = True

    def on_operation_start(self, event):
        self.running_operations.add(event.name)

    def on_operation_end(self, event):
        self.running_operations.remove(event.name)
        self.completed_operations.add(event.name)

    def on_task_end(self, event):
        self.completed_tasks[event.name] += event.num_tasks
        self.progress[event.name] += event.num_tasks / self.num_tasks[event.name]


def run_dash(app, *args, **kwargs):
    app.run(*args, **kwargs)


def run_dash_in_background_thread(app, *args, **kwargs):
    threading.Thread(
        target=run_dash, args=(app,) + args, kwargs=kwargs, daemon=False
    ).start()


def plan_to_cytoscape(
    plan,
    rankdir="TB",
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
    array_display_names = {}
    for _, d in dag.nodes(data=True):
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
                label += f"\ntasks: {int_repr(num_tasks)}"

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

    stylesheet = [
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

    layout = {
        "name": "dagre",
        "rankDir": rankdir,
        "rankSep": 36,
        "nodeDimensionsIncludeLabels": True,
        "fit": False,
    }
    cyto = dash_cytoscape.Cytoscape(
        id="cytoscape-component",
        layout=layout,
        # specify width and height here as dash cytoscape will set defaults that override css
        style={"width": "100%", "height": "100vh"},
        stylesheet=stylesheet,
        elements=elements,
        autoungrabify=True,
        minZoom=0.1,
        maxZoom=3,
    )

    return cyto, layout, array_display_names


def ops_to_arrays(dag):
    """Return a map from op name to the names of the arrays it produces"""
    op_name_to_array_names = {}
    for n, d in dag.nodes(data=True):
        node_type = d.get("type", None)
        if node_type == "op" and "primitive_op" in d:
            op_name_to_array_names[n] = list(successors_unordered(dag, n))

    return op_name_to_array_names


def plan_to_html(plan):
    return html.Div(
        children=[
            html.H3("Plan"),
            html.Table(
                [tr("Stages", int_repr(plan.num_stages))]
                + [tr("Operations", int_repr(plan.num_primitive_ops))]
                + [tr("Tasks", int_repr(plan.num_tasks))]
                + [tr("Allowed memory", memory_repr(plan.allowed_mem))]
                + [tr("Max projected memory", memory_repr(plan.max_projected_mem))]
                + [tr("Optimized", str(plan.optimized))]
            ),
            html.H3("Storage"),
            html.Table(
                [tr_header("", "Arrays", "Bytes", "Chunks")]
                + [
                    tr(
                        "Input",
                        int_repr(plan.total_input_narrays),
                        memory_repr(plan.total_input_nbytes),
                        int_repr(plan.total_input_nchunks),
                    )
                ]
                + [
                    tr(
                        "Intermediate",
                        int_repr(plan.total_intermediate_narrays),
                        memory_repr(plan.total_intermediate_nbytes),
                        int_repr(plan.total_intermediate_nchunks),
                    )
                ]
                + [
                    tr(
                        "Output",
                        int_repr(plan.total_output_narrays),
                        memory_repr(plan.total_output_nbytes),
                        int_repr(plan.total_output_nchunks),
                    )
                ]
                + [
                    tr(
                        "Total",
                        int_repr(plan.total_narrays),
                        memory_repr(plan.total_nbytes),
                        int_repr(plan.total_nchunks),
                    )
                ]
            ),
            html.H3("IO"),
            html.Table(
                [tr_header("", "Arrays", "Bytes", "Chunks")]
                + [
                    tr(
                        "Read",
                        int_repr(plan.total_narrays_read),
                        memory_repr(plan.total_nbytes_read),
                        int_repr(plan.total_nchunks_read),
                    )
                ]
                + [
                    tr(
                        "Write",
                        int_repr(plan.total_narrays_written),
                        memory_repr(plan.total_nbytes_written),
                        int_repr(plan.total_nchunks_written),
                    )
                ]
            ),
        ],
        className="info-panel",
    )


def op_to_html(data):
    children = [tr("Name", data["name"])]
    children.append(tr("Operation", data["op_name"]))

    if "primitive_op" in data:
        primitive_op = data["primitive_op"]
        children.append(tr("Projected memory", memory_repr(primitive_op.projected_mem)))
        children.append(tr("Tasks", int_repr(primitive_op.num_tasks)))
        if primitive_op.write_chunks is not None:
            children.append(tr("Write chunk shape", str(primitive_op.write_chunks)))

    if "pipeline" in data:
        pipeline = data["pipeline"]
        if isinstance(pipeline.config, BlockwiseSpec):
            children.append(
                tr("Num input blocks", str(pipeline.config.num_input_blocks))
            )
            children.append(
                tr("Num output blocks", str(pipeline.config.num_output_blocks))
            )

    if "stack_summaries" in data and data["stack_summaries"] is not None:
        # add call stack information
        stack_summaries = data["stack_summaries"]

        first_cubed_i = min(i for i, s in enumerate(stack_summaries) if s.is_cubed())
        caller_summary = stack_summaries[first_cubed_i - 1]

        calls = " -> ".join(
            [s.name for s in stack_summaries if not s.is_on_python_lib_path()]
        )

        line = f"{caller_summary.lineno} in {caller_summary.name}"

        # use title to set tooltip for long line
        children.append(html.Tr([html.Td("Calls"), html.Td(calls, title=calls)]))
        children.append(tr("Line", line))

    return html.Div(children=[html.Table(children=children)])


def array_to_html(data, array_display_names, plan):
    target = data["target"]

    name = data["name"]
    children = [tr("Name", data["name"])]
    if name in array_display_names:
        children.append(tr("Variable name", array_display_names[name]))

    children.append(tr("Shape", str(target.shape)))
    children.append(tr("Chunk shape", str(target.chunks)))
    children.append(tr("Data type", str(target.dtype)))
    children.append(tr("Chunk memory", memory_repr(chunk_memory(target))))
    if hasattr(target, "nbytes"):
        children.append(tr("Bytes", memory_repr(target.nbytes)))
    if hasattr(target, "nchunks"):
        children.append(tr("Chunks", int_repr(target.nchunks)))

    children.append(tr("Role", str(plan.array_role(name).value)))
    children.append(tr("Stored", str(hasattr(target, "store"))))
    if hasattr(target, "store"):
        # Use title to set tooltip for long line
        children.append(
            html.Tr(
                [html.Td("Store"), html.Td(str(target.store), title=str(target.store))]
            )
        )

    svg = array_to_svg(target)

    return html.Div(children=[html.Table(children=children), svg])


def array_to_svg(array):
    from cubed.vendor.dask.array.svg import svg

    chunks = normalize_chunks(array.chunks, shape=array.shape, dtype=array.dtype)
    s = svg(chunks, size=250)

    from xml.dom.minidom import Node, parseString

    document = parseString(s)
    svg_element = document.getElementsByTagName("svg")[0]

    def _extract_style(el):
        if not el.hasAttribute("style"):
            return None
        return {
            k.strip(): v.strip()
            for (k, v) in [x.split(":") for x in el.getAttribute("style").split(";")]
        }

    def handle_svg(svg):
        width = svg.getAttribute("width")
        height = svg.getAttribute("height")
        children = []
        for child in svg.childNodes:
            if child.nodeType == Node.ELEMENT_NODE:
                if child.tagName == "line":
                    children.append(handle_line(child))
                elif child.tagName == "polygon":
                    children.append(handle_polygon(child))
                elif child.tagName == "text":
                    children.append(handle_text(child))
        # convert width and height to a viewbox so that the image scales to the space available
        return dsvg.Svg(
            children, viewBox=f"0 0 {width} {height}", style=_extract_style(svg)
        )

    def handle_line(line):
        x1 = line.getAttribute("x1")
        y1 = line.getAttribute("y1")
        x2 = line.getAttribute("x2")
        y2 = line.getAttribute("y2")
        return dsvg.Line(x1=x1, y1=y1, x2=x2, y2=y2, style=_extract_style(line))

    def handle_polygon(polygon):
        points = polygon.getAttribute("points")
        return dsvg.Polygon(points=points, style=_extract_style(polygon))

    def handle_text(text):
        x = text.getAttribute("x")
        y = text.getAttribute("y")
        fontSize = text.getAttribute("font-size")
        fontWeight = text.getAttribute("font-weight")
        textAnchor = text.getAttribute("text-anchor")
        transform = text.getAttribute("transform")
        value = text.childNodes[0].data
        return dsvg.Text(
            [value],
            x=x,
            y=y,
            fontSize=fontSize,
            fontWeight=fontWeight,
            textAnchor=textAnchor,
            transform=transform,
            style=_extract_style(text),
        )

    return handle_svg(svg_element)


def tr_header(*ths):
    return html.Tr([html.Th(th) for th in ths])


def tr(*tds):
    return html.Tr([html.Td(td) for td in tds])


def int_repr(value):
    return f"{value:,}"
