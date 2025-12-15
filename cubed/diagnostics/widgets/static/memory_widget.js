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
                            "values": [{"allowed_mem": allowed_mem}]
                        },
                        "mark": "rule",
                        "encoding": {
                            "y": { "field": "allowed_mem", "type": "quantitative" },
                            "color": {
                                "datum": "allowed_mem",
                                // specify the legend colours here
                                "scale": {
                                    "range": ["#e06666", "#a9a9a9", "green"],
                                },
                                // "sort": ["allowed_mem", "projected_mem", "reserved_mem"],
                            },
                        },

                    },
                    {

                        "data": {
                            "values": [{"reserved_mem": reserved_mem}]
                        },
                        "mark": "rule",
                        "encoding": {
                            "y": { "field": "reserved_mem", "type": "quantitative" },
                            "color": {
                                "datum": "reserved_mem",
                            },
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
                        "layer": [
                            {
                                "mark": "bar",
                                "encoding": {
                                    "x": { "field": "x", "bin": { "binned": true }, "scale": { "domain": xDomain } },
                                    "x2": { "field": "x2" },
                                    "y": { "field": "projected_mem", "type": "quantitative" },
                                    "opacity": { "value": 0.2 },
                                    "tooltip": { "field": "name", "type": "nominal" }
                                }
                            },
                            {
                                "mark": "rule",
                                "encoding": {
                                    "x": { "field": "x", "type": "quantitative", "scale": { "domain": xDomain } },
                                    "x2": { "field": "x2", "type": "quantitative" },
                                    "y": { "field": "projected_mem", "type": "quantitative" },
                                    "color": {
                                        "datum": "projected_mem",
                                    }
                                }
                            },
                        ],
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
                                    { "test": `datum['actual_mem'] > ${allowed_mem}`, "value": "#e06666" },
                                    { "test": "datum['actual_mem'] > datum['projected_mem']", "value": "#f6b26b" },
                                ],
                                "value": "#9fc5e8",
                            },
                            "tooltip": { "field": "name", "type": "nominal" }
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
