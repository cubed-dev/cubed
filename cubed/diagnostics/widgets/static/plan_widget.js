import cytoscape from 'https://cdn.jsdelivr.net/npm/cytoscape@3.33.1/+esm'
import dagre from "https://esm.sh/cytoscape-dagre";

cytoscape.use(dagre);

const INITIALIZED_ARRAY_BACKGROUND_COLOR = "white";
const STORED_ARRAY_BACKGROUND_COLOR = "#aaffc3";

function render({ model, el }) {
    const div = document.createElement("div");
    div.classList.add("plan-widget");
    el.appendChild(div);

    const sidebar = document.createElement("div");
    sidebar.classList.add("plan-widget-sidebar");
    div.appendChild(sidebar);

    const summary = document.createElement("div");
    summary.classList.add("plan-widget-summary");
    sidebar.appendChild(summary);

    const details = document.createElement("div");
    details.classList.add("plan-widget-details");
    sidebar.appendChild(details);

    const graph = document.createElement("div");
    graph.style.height = model.get("height");
    model.on("change:height", () => {
        graph.style.height = model.get("height");
    });
    graph.classList.add("plan-widget-graph");
    div.appendChild(graph);

    if (model.get("_cytoscape_elements") && model.get("_cytoscape_elements").length > 0) {
        update_plan();
    } else {
        summary.innerHTML = "Plan widget waiting for computation to start..."
    }

    function update_plan() {
        summary.innerHTML = model.get("_summary_html");

        const _node_html_dict = model.get("_node_html_dict");
        function update_node_details() {
            const node_html = _node_html_dict[model.get("selected_node")];
            if (node_html) {
                details.innerHTML = node_html;
            } else {
                details.innerHTML = "<h3>Details</h3><p>Click on a node in the graph</p>";
            }
        }
        update_node_details();
        model.on("change:selected_node", update_node_details);

        const cy = cytoscape({
            container: graph,
            elements: model.get("_cytoscape_elements"),
            style: model.get("_cytoscape_style"),
            layout: model.get("_cytoscape_layout"),
            ...model.get("_cytoscape_options"),
        });

        const reset = document.createElement("button");
        reset.innerHTML = "Reset";
        reset.addEventListener("click", () => {
            cy.reset();
            cy.resize();  // https://stackoverflow.com/a/23484505
        });

        const fit = document.createElement("button");
        fit.innerHTML = "Fit";
        fit.addEventListener("click", () => {
            cy.fit();
            cy.resize();  // https://stackoverflow.com/a/23484505
        });

        const center = document.createElement("button");
        center.innerHTML = "Center";
        center.addEventListener("click", () => {
            cy.center();
            cy.resize();  // https://stackoverflow.com/a/23484505
        });

        const buttons = document.createElement("div");
        buttons.appendChild(reset);
        buttons.appendChild(fit);
        buttons.appendChild(center);
        buttons.classList.add("plan-widget-graph-controls");
        el.appendChild(buttons);

        // https://stackoverflow.com/a/23484505
        // https://github.com/plotly/dash-cytoscape/issues/212
        cy.on("tapstart", evt => {
            cy.resize();
        });

        cy.on("tap", evt => {
            if (evt.target === cy) {
                // tap on background - deselect
                model.set("selected_node", "");
                model.save_changes();
            } else {
                const node = evt.target;
                model.set("selected_node", node.id());
                model.save_changes();
            }
        });

        const runningOperations = new Set();

        model.on("msg:custom", msg => {
            if (msg.type === "on_task_end") {
                const pct = Math.round(msg.progress * 100);

                for (const array_name of msg.array_names) {
                    cy.style().selector(`node[id='${array_name}']`).style({
                        "background-fill": "linear-gradient",
                        "background-gradient-stop-colors": `${STORED_ARRAY_BACKGROUND_COLOR} ${INITIALIZED_ARRAY_BACKGROUND_COLOR}`,
                        "background-gradient-stop-positions": `${pct}% ${pct}%`,
                        "background-gradient-direction": "to-right",
                    }).update();
                }
            } else if (msg.type == "on_operation_start") {
                runningOperations.add(msg.op_name);

                // from https://songhaifan.github.io/learning_cytospace/#/chapters/4/?id=performance-optimized-animations
                let frame = 0;
                const animate = () => {
                    const progress = frame / 15;
                    cy.nodes(`node[id='${msg.op_name}']`).style("background-opacity", (Math.sin(progress) + 1) / 2);
                    frame++;
                    if (runningOperations.has(msg.op_name)) {
                        requestAnimationFrame(animate);
                    } else {
                        cy.nodes(`node[id='${msg.op_name}']`).style({
                            "background-opacity": 1.0
                        });
                    }
                };
                requestAnimationFrame(animate);

            } else if (msg.type == "on_operation_end") {
                runningOperations.delete(msg.op_name);
            }
        });
    }

    model.on("msg:custom", msg => {
        if (msg.type === "on_compute_start") {
            update_plan();
        }
    });
}
export default { render };
