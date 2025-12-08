import { vega } from 'https://cdn.jsdelivr.net/npm/vega-embed@7.1.0/+esm';
import vegaEmbed from 'https://cdn.jsdelivr.net/npm/vega-embed@7.1.0/+esm';

function render({ model, el }) {
    const div = document.createElement("div");
    div.style.width = model.get("width");
    div.style.height = model.get("height");
    el.classList.add("vega-vis");
    el.appendChild(div);

    // TODO: add holding text "Timeline widget waiting for computation to start..."

    model.on("msg:custom", msg => {
        if (msg.type === "on_compute_start") {

            const vlSpec = {
                "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
                "autosize": {
                    "type": "fit",
                    "resize": true,
                    "contains": "padding"
                },
                "width": "container",
                "height": "container",
                "description": "Timeline",
                "data": {
                    name: "data",
                    values: [],
                },
                "mark": {"type": "point", "filled": true},
                "encoding": {
                    "x": { "field": "elapsed_time", "type": "quantitative", "title": "Execution time (s)", },
                    "y": { "field": "task_index", "type": "quantitative", "title": "Task number" },
                    "color": {
                        "field": "event_type",
                        "type": "nominal",
                        "title": null,
                        "legend": { "orient": "top-left" },
                        "scale": { "range": ["red", "orange", "blue", "green"], },
                        "sort": ["task create", "function start", "function end", "task_result"],
                    },
                }
            };
            vegaEmbed(div, vlSpec).then((res) => {
                model.on("msg:custom", msg => {
                    if (msg.type === "on_task_end") {
                        const cs = vega.changeset();
                        cs.insert(msg.values);
                        res.view.change("data", cs).run();
                    }
                });
            });

        }
    });


}
export default { render };
