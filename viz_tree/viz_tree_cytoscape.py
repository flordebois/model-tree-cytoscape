from __future__ import annotations
from typing import Any
from viz_tree.viz_tree import VizTree
from viz_tree.nodes import CombinedLinNode, InternalNode, LeafNode
import numpy as np
import matplotlib.pyplot as plt
from viz_tree.predsplot import predsplot

# TODO: fix colors
cmap = plt.cm.tab10.colors
feature_colors = [cmap[3], cmap[1], cmap[9], cmap[2], "grey", "grey", cmap[5]]


# ── Convert a VizTree into Cytoscape elements (nodes + edges) ──────────────────
def to_cytoscape_elements(
        viz_tree: "VizTree",
        combine_lin: bool = False,
        use_regplots: bool = False,
        use_predsplots: bool = False,
) -> list[dict[str, Any]]:
    nodes = viz_tree.nodes
    edges = viz_tree.edges

    # ── combine consecutive lin nodes ──────────────────────────────────────────
    if combine_lin:  # Same code as in viz_tree.py method .get_dot(...)
        new_mapping: dict = {}
        for node in nodes:
            new_mapping[node] = node

        combined_nodes = []
        for node in nodes:
            if (
                    node.type == "lin"
                    and node.left_child_node.type == "lin"
                    and new_mapping[node] is node
            ):
                chain = [node]
                next_node = node.left_child_node
                while next_node.type == "lin":
                    chain.append(next_node)
                    next_node = next_node.left_child_node
                combined_lin_node = CombinedLinNode(chain)
                combined_lin_node.set_id(chain[0].id)
                combined_nodes.append(combined_lin_node)
                for node_chain in chain:
                    new_mapping[node_chain] = combined_lin_node
            elif node.type == "lin" and new_mapping[node] is not node:
                continue  # Already combined linear nodes
            else:
                combined_nodes.append(node)  # Other nodes

        combined_edges = []
        for parent, child in edges:
            new_parent_node = new_mapping[parent]
            new_child_node = new_mapping[child]
            # Don't add edges that connect combined linear nodes
            if new_parent_node is not new_child_node:
                combined_edges.append((new_parent_node, new_child_node))

        nodes = combined_nodes
        edges = combined_edges

    # ── Build node elements ────────────────────────────────────────────────────
    elements: list[dict[str, Any]] = []
    node_to_id: dict = {}

    for node in nodes:
        node_type = node.type
        label = node.get_label()
        n_samples = int(node.X.shape[0])

        data: dict[str, Any] = {
            "id": f"node{node.id}",
            "node_type": node_type,
            "label": label,
            "n_samples": n_samples,
        }

        regplot_class = ""
        if use_regplots and isinstance(node, InternalNode):
            directory_regplot_file = node.make_regression_plot(
                "/Users/flor/Pycharm/PILOT-VIS/scripts/output/live/regplots",
                (5, 3),
                feature_colors,
                None,
                None)
            data["dir_regplot"] = f"/internal_regplots/regplot_node{node.id}.svg"
            regplot_class = " regplot"

        predsplot_class = ""
        if use_predsplots and isinstance(node, LeafNode):
            predsplot(node.X, node.coefficients, y_hat=np.sum(node.coefficients * node.X, axis=1) + node.intercept,
                      n_max=5, intercept=None, fig_size=(5, 3), feature_names=None, all_feature_colors=feature_colors,
                      display_type="histogram", truncate_total_pred=False, variable_tick_width=True,
                      file_directory=f"/Users/flor/Pycharm/PILOT-VIS/scripts/output/live/predsplots/predsplot_node{node.id}.svg",
                      highlight_x=None, staircase=False)
            data["dir_predsplot"] = f"/internal_predsplots/predsplot_node{node.id}.svg"
            predsplot_class = " predsplot"

        elements.append({"data": data, "classes": node_type + regplot_class + predsplot_class})

    # ── Build edge elements ────────────────────────────────────────────────────
    for parent, child in edges:
        edge_data: dict[str, Any] = {"source": f"node{parent.id}", "target": f"node{child.id}"}
        elements.append({"data": edge_data})

    return elements


# ── Stylesheet builder ─────────────────────────────────────────────────────────
def build_cytoscape_stylesheet() -> list[dict]:
    font_family = "Arial, sans-serif"
    font_size = 12
    selected_color = "#000000"
    base = [
        # ── Default node ──────────────────────────────────────────────────────
        {
            "selector": "node",
            "style": {
                "label": "data(label)",
                "text-valign": "center",
                "text-halign": "center",
                "text-wrap": "wrap",
                "text-max-width": "140px",
                "font-family": font_family,
                "font-size": f"{font_size}px",
                "color": "#ffffff",
                "background-color": "#000000",  # default, normally not used
                "shape": "rectangle",  # default, normally not used
                "width": "100px",
                "height": "40px",
                "padding": "6px",
                "border-width": "1.5px",
                "border-color": "#000000",
            },
        },
        # ── Default edge ──────────────────────────────────────────────────────
        {
            "selector": "edge",
            "style": {
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "target-arrow-color": "#000000",
                "line-color": "#000000",
                "width": 1.5,
                "label": "data(label)",
                "font-family": font_family,
                "font-size": f"{font_size}px",
                "color": "#000000",
                "text-background-color": "#ffffff",
                "text-background-opacity": 0.7,
                "text-background-padding": "2px",
            },
        },
        # ── specific nodes ────────────────────────────────────────────────────
        {
            "selector": "node.leaf",
            "style": {
                "background-color": "#2ca02c",
                "shape": "ellipse",
                "width": "160px",
                "height": "60px",
            },
        },
        {
            "selector": "node.lin",
            "style": {
                "background-color": "#9467bd",
                "shape": "ellipse",
            },
        },
        {
            "selector": "node.blin",
            "style": {
                "background-color": "#1f77b4",
                "shape": "roundrectangle",
            },
        },
        {
            "selector": "node.pcon",
            "style": {
                "background-color": "#d62728",
                "shape": "roundrectangle",
            },
        },
        {
            "selector": "node.plin",
            "style": {
                "background-color": "#ff7f0e",
                "shape": "roundrectangle",
            },
        },
        {
            "selector": "node.pconc",
            "style": {
                "background-color": "#8c564b",
                "shape": "roundrectangle",
            },
        },
        {
            "selector": "node.combined_lin",
            "style": {
                "background-color": "#9467bd",
                "shape": "ellipse",
            },
        },
        {
            "selector": "node.regplot",
            "style": {
                "label": "",
                "background-image": 'data(dir_regplot)',
                "shape": "rectangle",
                "width": "500px",
                "height": "300px",
            },
        },
        {
            "selector": "node.predsplot",
            "style": {
                "label": "",
                "background-image": 'data(dir_predsplot)',
                "shape": "rectangle",
                "width": "500px",
                "height": "300px",
            },
        },
        # ── Selected highlight ────────────────────────────────────────────────
        {
            "selector": "node:selected",
            "style": {
                "border-width": "3px",
                "border-color": "#f0e442",
            },
        },
        # ── Highlighted path (add class "on-path" via callback) ───────────────
        {
            "selector": "node.on-path",
            "style": {
                "border-width": "5px",
                "border-color": selected_color,
                "border-opacity": 1,
            },
        },
        {
            "selector": "edge.on-path",
            "style": {
                "line-color": selected_color,
                "target-arrow-color": selected_color,
                "width": 4,
            },
        },
    ]
    return base


# ── Path-highlighting helper ───────────────────────────────────────────────────

def highlight_path_for_x(
        viz_tree: "VizTree",
        x_sample: np.ndarray,
        elements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Given an input sample `x_sample` (1-D array-like), walk the tree and add
    the class "on-path" to every node and edge along the prediction path.

    Returns a *new* elements list (does not mutate the original).
    """

    # Collect node/edge ids on the prediction path
    path_node_ids: set[str] = set()
    path_edge_pairs: set[tuple[str, str]] = set()

    # Walk the original viz_tree nodes list so indices match

    current = viz_tree.root_node
    path_node_ids.add(f"node{current.id}")

    while current.type != "leaf":
        parent_id = current.id
        if current.type == "lin":
            current = current.left_child_node
        else:
            if x_sample[current.pivot_idx] > current.pivot_value:
                current = current.right_child_node
            else:
                current = current.left_child_node

        path_node_ids.add(f"node{current.id}")
        path_edge_pairs.add((f"node{parent_id}", f"node{current.id}"))

    # Return a copy of elements with updated classes
    new_elements = []
    for element in elements:
        element_copy = {**element, "data": {**element["data"]}}
        classes = element_copy.get("classes", "")
        class_set = set(classes.split())

        # Remove stale on-path
        class_set.discard("on-path")

        if "source" not in element["data"]:
            # It's a node
            if element["data"]["id"] in path_node_ids:
                class_set.add("on-path")
        else:
            # It's an edge
            src = element["data"]["source"]
            tgt = element["data"]["target"]
            if (src, tgt) in path_edge_pairs:
                class_set.add("on-path")

        element_copy["classes"] = " ".join(class_set)
        new_elements.append(element_copy)

    return new_elements
