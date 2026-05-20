from __future__ import annotations
from typing import Any
from viz_tree.viz_tree import VizTree
from viz_tree.nodes import CombinedLinNode, InternalNode, LeafNode
import numpy as np
import matplotlib.pyplot as plt
from viz_tree.predsplot import predsplot
from viz_tree.regplot import make_regression_plot
from datetime import datetime


# ── Convert a VizTree into Cytoscape elements (nodes + edges) ──────────────────
def to_cytoscape_elements(
        viz_tree: "VizTree",
        combine_lin: int = 1, # 1 - don't combine, 2 - combine to node, 3 - combine to edge
        use_regplots: bool = False,
        use_predsplots: bool = False,
        fig_size = (5, 3),
        predsplot_n_max = 5,
        predsplot_use_intercept = False,
        predsplot_display_type = "histogram",
        predsplot_truncate_total_pred = True,
        predsplot_staircase = False,
        highlight_x = None,
        only_show_highlight = False,
) -> list[dict[str, Any]]:
    nodes = viz_tree.nodes
    edges = viz_tree.edges

    # ── Find nodes to highlight ──────────────────────────────────────────
    highlight_nodes = []
    highlight_edges = []
    if highlight_x is not None:
        highlight_nodes.append(viz_tree.root_node)
        node_in_path = viz_tree.root_node
        while node_in_path.type != "leaf":
            parent_node_in_path = node_in_path
            if node_in_path.type == "lin":
                node_in_path = node_in_path.left_child_node
            else:
                if highlight_x[node_in_path.pivot_idx] > node_in_path.pivot_value:
                    node_in_path = node_in_path.right_child_node
                else:
                    node_in_path = node_in_path.left_child_node
            highlight_nodes.append(node_in_path)
            highlight_edges.append((parent_node_in_path, node_in_path))

        if only_show_highlight:
            nodes = highlight_nodes
            edges = highlight_edges

    # ── combine consecutive lin nodes ──────────────────────────────────────────
    combined_lin_edges = []
    if combine_lin != 0:
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

                if combine_lin == 1:
                    combined_lin_node = CombinedLinNode(chain)
                    combined_lin_node.set_id(chain[0].id)
                    combined_nodes.append(combined_lin_node)
                    for node_chain in chain:
                        new_mapping[node_chain] = combined_lin_node
                    if node in highlight_nodes:
                        highlight_nodes.append(combined_lin_node)
                elif combine_lin == 2:
                    for node_chain in chain:
                        new_mapping[node_chain] = chain[-1].left_child_node

            elif node.type == "lin" and new_mapping[node] is not node:
                continue  # Already combined linear nodes
            elif combine_lin == 2 and node.type == "lin":
                new_mapping[node] = node.left_child_node
            else:
                combined_nodes.append(node)  # Other nodes

        combined_edges = []
        for parent, child in edges:
            new_parent_node = new_mapping[parent]
            new_child_node = new_mapping[child]
            # Don't add edges that connect combined linear nodes
            if new_parent_node is not None and new_parent_node is not new_child_node:
                combined_edges.append((new_parent_node, new_child_node))
                if (parent, child) in highlight_edges:
                    highlight_edges.append((new_parent_node, new_child_node))
                if combine_lin == 2 and child != new_child_node:
                    combined_lin_edges.append((new_parent_node, new_child_node))

        nodes = combined_nodes
        edges = combined_edges

    # ── Build node elements ────────────────────────────────────────────────────
    elements: list[dict[str, Any]] = []
    elements_id  = datetime.now().strftime('%d-%m-%y_%H-%M-%S')

    for node in nodes:
        node_type = node.type
        label = node.get_label()
        n_samples = np.sum(node.indices)

        data: dict[str, Any] = {
            "id": f"node{node.id}",
            "node_type": node_type,
            "label": label,
            "label_minimal": node.get_minimal_label(),
            "n_samples": n_samples,
        }
        classes  = [node_type]
        node_highlight_x = None
        if node in highlight_nodes:
            classes.append("highlight")
            node_highlight_x = highlight_x

        if use_regplots and isinstance(node, InternalNode):
            make_regression_plot(
                node,
                viz_tree.X_train,
                f"/Users/flor/Pycharm/PILOT-VIS/scripts/output/live/regplots/regplot_node{node.id}_{elements_id}.svg",
                fig_size,
                viz_tree.feature_colors,
                None,
                node_highlight_x
            )
            data["dir_regplot"] = f"/internal_regplots/regplot_node{node.id}_{elements_id}.svg"
            classes.append("regplot")

        if use_predsplots and isinstance(node, LeafNode):
            if np.any(node.coefficients != 0):
                X = viz_tree.X_train[node.indices, :]
                if predsplot_use_intercept:
                    intercept = node.intercept
                else:
                    intercept = None
                predsplot(X, node.coefficients, y_hat=np.sum(node.coefficients * X, axis=1) + node.intercept,
                          n_max=predsplot_n_max, intercept=intercept, fig_size=fig_size, feature_names=None,
                          all_feature_colors=viz_tree.feature_colors,
                          display_type=predsplot_display_type, truncate_total_pred=predsplot_truncate_total_pred, variable_tick_width=True,
                          file_directory=f"/Users/flor/Pycharm/PILOT-VIS/scripts/output/live/predsplots/predsplot_node{node.id}_{elements_id}.svg",
                          highlight_x=node_highlight_x, staircase=predsplot_staircase)
                data["dir_predsplot"] = f"/internal_predsplots/predsplot_node{node.id}_{elements_id}.svg"
                classes.append("predsplot")


        elements.append({"data": data, "classes": " ".join(classes)})

    # ── Build edge elements ────────────────────────────────────────────────────
    for parent, child in edges:
        edge_data: dict[str, Any] = {"source": f"node{parent.id}", "target": f"node{child.id}"}
        classes = []
        if (parent, child) in highlight_edges:
            classes.append("highlight")
        if (parent, child) in combined_lin_edges:
            classes.append("combine_lin")
        elements.append({"data": edge_data, "classes": " ".join(classes)})

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
            "selector": "node.collapsed",
            "style": {
                "background-color": "#808080",
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
        {
            "selector": "node.minimal",
            "style": {
                "label": "data(label_minimal)",
                "width": "30px",
                "height": "15px",
            },
        },
        # ── specific edges ────────────────────────────────────────────────────
        {
            "selector": "edge.combine_lin",
            "style": {
                "mid-target-arrow-shape": "circle",
                "mid-target-arrow-color": "#9467bd",
            }
        },

        # ── Selected highlight ────────────────────────────────────────────────
        {
            "selector": "node:selected",
            "style": {
                "border-width": "3px",
                "border-color": "#f0e442",
            },
        },
        # ── Highlighted path  ────────────────────────────────────────────────
        {
            "selector": "node.highlight",
            "style": {
                "border-width": "5px",
                "border-color": selected_color,
                "border-opacity": 1,
            },
        },
        {
            "selector": "edge.highlight",
            "style": {
                "line-color": selected_color,
                "target-arrow-color": selected_color,
                "width": 4,
            },
        },
    ]
    return base
