import { vega } from 'https://cdn.jsdelivr.net/npm/vega-embed@7.1.0/+esm';
import vegaEmbed from 'https://cdn.jsdelivr.net/npm/vega-embed@7.1.0/+esm';

function render({ model, el }) {
    const div = document.createElement("div");
    div.style.width = model.get("width");
    div.style.height = model.get("height");
    el.classList.add("vega-vis");
    el.appendChild(div);

    div.innerHTML = "Memory widget waiting for computation to start..."

    model.on("msg:custom", msg => {
        if (msg.type === "on_compute_start") {
            div.innerHTML = "";

            const num_tasks = model.get("_num_tasks");
            const allowed_mem = model.get("_allowed_mem");
            const reserved_mem = model.get("_reserved_mem");
            const projected_mem_by_op = model.get("_projected_mem_by_op");
            const xDomain = [-0.5, num_tasks - 0.5];
            const vlSpec = {
                "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
                "autosize": {
                    "type": "fit",
                    "contains": "padding"
                },
                "width": "container",
                "description": "Task memory usage",

                "layer": [
                    {

                        "data": {
                            "values": projected_mem_by_op,
                        },
                        "transform": [
                            { "calculate": "datum.start_task_index - 0.5", "as": "x" },
                            { "calculate": "datum.end_task_index - 0.5", "as": "x2" }
                        ],
                        "mark": "bar",
                        "encoding": {
                            "x": { "field": "x", "bin": { "binned": true }, "scale": { "domain": xDomain } },
                            "x2": { "field": "x2" },
                            "y": { "field": "projected_mem", "type": "quantitative" },
                            "color": {
                                "value": "#dcbeff",
                            },
                            "tooltip": [{ "field": "name", "type": "nominal", "title": "Operation" }]
                        }

                    },
                    {

                        "data": {
                            "values": [{ "allowed_mem": allowed_mem }]
                        },
                        "mark": "rule",
                        "encoding": {
                            "y": { "field": "allowed_mem", "type": "quantitative" },
                            "color": {
                                "datum": "allowed_mem",
                                // specify the legend colours here (alphabetical order: allowed, reserved, projected)
                                "scale": {
                                    "range": ["#e6194B", "#a9a9a9", "#3cb44b"],
                                },
                            },
                            "strokeWidth": { "value": 3 },
                            "tooltip": [{ "field": "allowed_mem", "type": "quantitative", "title": "Allowed mem (MB)" }],
                        },
                    },
                    {

                        "data": {
                            "values": [{ "reserved_mem": reserved_mem }]
                        },
                        "mark": "rule",
                        "encoding": {
                            "y": { "field": "reserved_mem", "type": "quantitative" },
                            "color": {
                                "datum": "reserved_mem",
                            },
                            "strokeWidth": { "value": 3 },
                            "tooltip": [{ "field": "reserved_mem", "type": "quantitative", "title": "Reserved mem (MB)" }],
                        },

                    },
                    {

                        "data": {
                            "values": projected_mem_by_op,
                        },
                        "transform": [
                            { "calculate": "datum.start_task_index - 0.5", "as": "x" },
                            { "calculate": "datum.end_task_index - 0.5", "as": "x2" }
                        ],
                        "mark": "rule",
                        "encoding": {
                            "x": { "field": "x", "type": "quantitative", "scale": { "domain": xDomain } },
                            "x2": { "field": "x2", "type": "quantitative" },
                            "y": { "field": "projected_mem", "type": "quantitative" },
                            "color": {
                                "datum": "projected_mem",
                            },
                            "strokeWidth": { "value": 3 },
                            "tooltip": [{ "field": "projected_mem", "type": "quantitative", "title": "Projected mem (MB)" }],
                        }

                    },
                    {

                        "data": {
                            name: "data",
                            values: [],
                        },
                        "mark": "bar",
                        "encoding": {
                            "x": { "field": "task_index", "type": "quantitative", "scale": { "domain": xDomain }, "title": "Task number" },
                            "y": { "field": "actual_mem", "type": "quantitative", "title": "Task memory (MB)" },
                            "color": {
                                "condition": [
                                    { "test": `datum['actual_mem'] > ${allowed_mem}`, "value": "#e6194B" },
                                    { "test": "datum['actual_mem'] > datum['projected_mem']", "value": "#ffe119" },
                                ],
                                "value": "#4363d8",
                            },
                            "tooltip": [{ "field": "actual_mem", "type": "quantitative", "title": "Actual mem (MB)" }],
                        }
                    },
                ],

            };
            vegaEmbed(div, vlSpec).then((res) => {
                model.on("msg:custom", msg => {
                    if (msg.type === "on_task_end") {
                        const cs = vega.changeset();
                        cs.insert(msg);
                        res.view.change("data", cs).run();
                    }
                });
            });

        }
    });


}
export default { render };
