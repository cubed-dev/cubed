import inspect
import threading

import dash_cytoscape
from dash import Dash, Input, Output, callback, dcc, html

from cubed.runtime.pipeline import visit_nodes
from cubed.runtime.types import Callback
from cubed.storage.store import is_storage_array
from cubed.storage.zarr import LazyZarrArray
from cubed.utils import extract_stack_summaries

# Load dagre layout engine
dash_cytoscape.load_extra_layouts()


class DashPlan(Callback):
    def __init__(self, array, debug=True):
        app = self.create_dash_app(array)
        run_dash_in_background_thread(app, debug=debug)

    def create_dash_app(self, array):
        app = Dash()
        cyto = plan_to_cytoscape(array.plan, optimize_graph=False)

        self.default_stylesheet = cyto.stylesheet

        app.layout = html.Div(
            [
                cyto,
                dcc.Interval(id="interval-component", interval=200, n_intervals=0),
            ]
        )

        @callback(
            Output("cytoscape-component", "stylesheet"),
            Input("interval-component", "n_intervals"),
        )
        def update_graph_live(n):
            # set op node background fill opacity to progress value
            stylesheet = list(self.default_stylesheet)
            for name, progress in self.progress.items():
                stylesheet.append(
                    {
                        "selector": f"node[id = '{name}']",
                        "style": {
                            "background-opacity": f"{progress}",
                        },
                    }
                )
            return stylesheet

        return app

    def on_compute_start(self, event):
        self.num_tasks = {}
        self.progress = {}
        for name, node in visit_nodes(event.dag):
            self.num_tasks[name] = node["primitive_op"].num_tasks
            self.progress[name] = 0.0

    def on_task_end(self, event):
        self.progress[event.name] += event.num_tasks / self.num_tasks[event.name]


def run_dash(app, debug):
    app.run(debug=debug, use_reloader=False)


def run_dash_in_background_thread(app, debug=True):
    threading.Thread(target=run_dash, args=(app, debug), daemon=True).start()


def plan_to_cytoscape(
    plan,
    rankdir="TB",
    optimize_graph=True,
    optimize_function=None,
    show_hidden=False,
):
    finalized_plan = plan._finalize(optimize_graph, optimize_function)
    dag = finalized_plan.dag
    dag = dag.copy()  # make a copy since we mutate the DAG below

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
    print(stack_summaries)
    first_cubed_i = min(i for i, s in enumerate(stack_summaries) if s.is_cubed())
    caller_summary = stack_summaries[first_cubed_i - 1]
    array_display_names.update(caller_summary.array_names_to_variable_names)

    elements = []

    # now set node attributes with visualization info
    for n, d in dag.nodes(data=True):
        label = n
        node_type = d.get("type", None)
        if node_type == "op":
            label = d["op_display_name"]
            op_name = d["op_name"]
            if op_name == "blockwise":
                fillcolor = "#dcbeff"
            elif op_name == "rechunk":
                fillcolor = "#aaffc3"
            else:
                fillcolor = "white"

            num_tasks = None
            if "primitive_op" in d:
                primitive_op = d["primitive_op"]
                num_tasks = primitive_op.num_tasks

            if num_tasks is not None:
                label += f"\ntasks: {num_tasks}"

            elements.append(
                {
                    "data": {
                        "id": n,
                        "label": label,
                        "shape": "round-rectangle",
                        "fillcolor": fillcolor,
                    }
                }
            )

        elif node_type == "array":
            target = d["target"]

            # materialized arrays are light orange, virtual arrays are white
            if isinstance(target, LazyZarrArray) or is_storage_array(target):
                fillcolor = "#ffd8b1"
            else:
                fillcolor = "white"
            if n in array_display_names:
                var_name = array_display_names[n]
                label = f"{n}\n{var_name}"

            elements.append(
                {
                    "data": {
                        "id": n,
                        "label": label,
                        "shape": "rectangle",
                        "fillcolor": fillcolor,
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
                        "fillcolor": "white",
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
                "border-color": "black",
                "border-width": 1,
                "opacity": "1.0",
                "text-valign": "center",
                "text-halign": "center",
                "label": "data(label)",
                "shape": "data(shape)",
                "text-wrap": "wrap",
                # TODO: following is deprecated, see https://stackoverflow.com/a/78033670
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
    ]

    cyto = dash_cytoscape.Cytoscape(
        id="cytoscape-component",
        layout={
            "name": "dagre",
            "rankDir": rankdir,
            "rankSep": 36,
            "nodeDimensionsIncludeLabels": True,
            "fit": True,
        },
        style={"width": "100%", "height": "500px"},
        stylesheet=stylesheet,
        elements=elements,
    )

    return cyto
