from __future__ import annotations
from typing import Any

from nodes.collapsed_node import CollapsedNode
from nodes.split_node import PconcNode
from viz_tree.viz_tree import VizTree
from nodes.base_node import BaseNode
from nodes.leaf_node import LeafNode
from nodes.internal_node import InternalNode, LinearNode
from nodes.combined_lin_node import CombinedLinNode
from nodes.none_node import NoneNode
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from plots.predsplot import predsplot
from plots.predsplot2 import predsplot2
from plots.regplot import make_regression_plot
from datetime import datetime
from config import NODE_TYPE_COLORS

# ── Convert a VizTree into Cytoscape elements (nodes + edges) ──────────────────
def viz_tree_to_cytoscape_elements(
        viz_tree: "VizTree",
        dir_live: str,
        combine_lin: int = 0, # 0 - don't combine, 1 - combine to node, 2 - combine to edge
        show_rss: bool = False,
        use_color_features: bool = False,
        show_node_plots: bool = False,
        fig_size = (5, 3),
        predsplot_n_max = 5,
        predsplot_use_intercept = False,
        predsplot_display_type = "histogram",
        predsplot_truncate_total_pred = True,
        predsplot_staircase = False,
        predsplot_type2 = False,
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
        while not isinstance(node_in_path, (LeafNode, CollapsedNode, NoneNode)):
            parent_node_in_path = node_in_path
            if isinstance(node_in_path, LinearNode):
                node_in_path = node_in_path.child
            elif isinstance(node_in_path, PconcNode):
                if highlight_x[node_in_path.pivot_idx] in node_in_path.pivot_value:
                    node_in_path = node_in_path.left_child
                else:
                    node_in_path = node_in_path.right_child
            else:
                if highlight_x[node_in_path.pivot_idx] > node_in_path.pivot_value:
                    node_in_path = node_in_path.right_child
                else:
                    node_in_path = node_in_path.left_child
            highlight_nodes.append(node_in_path)
            highlight_edges.append((parent_node_in_path, node_in_path))

    # ── combine consecutive lin nodes ──────────────────────────────────────────
    combined_lin_edges = []
    if combine_lin != 0:
        new_mapping: dict = {}
        for node in nodes:
            new_mapping[node] = node

        combined_nodes = []
        for node in nodes:
            if (
                    isinstance(node, LinearNode)
                    and isinstance(node.child, LinearNode)
                    and new_mapping[node] is node
            ):
                chain = [node]
                next_node = node.child
                while isinstance(next_node, LinearNode):
                    chain.append(next_node)
                    next_node = next_node.child

                if combine_lin == 1:
                    combined_lin_node = CombinedLinNode(chain)
                    combined_nodes.append(combined_lin_node)
                    for node_chain in chain:
                        new_mapping[node_chain] = combined_lin_node
                    if node in highlight_nodes:
                        highlight_nodes.append(combined_lin_node)
                elif combine_lin == 2:
                    for node_chain in chain:
                        new_mapping[node_chain] = chain[-1].child

            elif isinstance(node, LinearNode) and new_mapping[node] is not node:
                continue  # Already combined linear nodes
            elif combine_lin == 2 and isinstance(node, LinearNode):
                new_mapping[node] = node.child
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

    if show_node_plots or use_color_features:
        n_features = viz_tree.X_train.shape[1]
        cmap = plt.colormaps["tab20"].resampled(n_features)
        feature_colors = [cmap(i) for i in range(n_features)]
        feature_colors_hex = [to_hex(color) for color in feature_colors]

    for node in nodes:
        if highlight_x is not None and only_show_highlight and node not in highlight_nodes:
            continue
        if isinstance(node, NoneNode):
            continue
        node_type = node.__class__.__name__
        label = node.get_label()
        if show_rss:
            label += f"\nrss:{node.rss:.5g}"
        n_samples = np.sum(node.indices)

        if use_color_features:
            if isinstance(node, InternalNode):
                color = feature_colors_hex[node.pivot_idx]
            else:
                color = NODE_TYPE_COLORS[node_type]
        else:
            color = NODE_TYPE_COLORS[node_type]

        data: dict[str, Any] = {
            "id": f"node{node.id}",
            "node_type": node_type,
            "color": color,
            "label": label,
            "label_minimal": node.get_minimal_label(),
            "n_samples": n_samples,
            "rss": float(f"{node.rss:.5g}"),
            "rss_root_reduction": float(f"{(1 - node.rss/viz_tree.root_node.rss)*100:.3g}"),
            "highlight": True if node in highlight_nodes else False,
        }
        classes  = [node_type]
        node_highlight_x = None
        if node in highlight_nodes:
            classes.append("highlight")
            node_highlight_x = highlight_x

        if show_node_plots and isinstance(node, InternalNode):
            make_regression_plot(
                node,
                viz_tree.X_train,
                f"{dir_live}/regplots/regplot_node{node.id}_{elements_id}.svg",
                fig_size,
                feature_colors,
                None,
                node_highlight_x
            )
            data["dir_regplot"] = f"/internal_regplots/regplot_node{node.id}_{elements_id}.svg"
            classes.append("regplot")

        if show_node_plots and isinstance(node, LeafNode):
            if predsplot_type2:
                predsplot2(viz_tree=viz_tree,
                           leaf_node=node,
                           y_hat = viz_tree.y_hat,
                           n_max=predsplot_n_max,
                           fig_size=fig_size,
                           truncate_total_pred=predsplot_truncate_total_pred,
                           variable_tick_width=True,
                           display_type=predsplot_display_type,
                           file_directory=f"{dir_live}/predsplots/predsplot2_node{node.id}_{elements_id}.svg",
                           highlight_x=node_highlight_x,
                           staircase=predsplot_staircase,
                           feature_names=None,
                           all_feature_colors=feature_colors,
                           )
                data["dir_predsplot"] = f"/internal_predsplots/predsplot2_node{node.id}_{elements_id}.svg"
                classes.append("predsplot")
            else:
                if np.any(np.array(node.node_model.coefficients) != 0):
                    node_X = viz_tree.X_train[node.indices, :]
                    if predsplot_use_intercept:
                        intercept = node.node_model.intercept
                    else:
                        intercept = None
                    predsplot(node_X,
                              np.array(node.node_model.coefficients),
                              y_hat=np.array(
                                  np.sum(node.node_model.coefficients * node_X, axis=1) + node.node_model.intercept),
                              n_max=predsplot_n_max,
                              intercept=intercept,
                              fig_size=fig_size,
                              feature_names=None,
                              all_feature_colors=feature_colors,
                              display_type=predsplot_display_type,
                              truncate_total_pred=predsplot_truncate_total_pred,
                              variable_tick_width=True,
                              file_directory=f"{dir_live}/predsplots/predsplot_node{node.id}_{elements_id}.svg",
                              highlight_x=node_highlight_x,
                              staircase=predsplot_staircase)
                    data["dir_predsplot"] = f"/internal_predsplots/predsplot_node{node.id}_{elements_id}.svg"
                    classes.append("predsplot")

        elements.append({"data": data, "classes": " ".join(classes)})

    # ── Build edge elements ────────────────────────────────────────────────────
    for parent, child in edges:
        if highlight_x is not None and only_show_highlight and child not in highlight_nodes:
            continue
        if isinstance(child, NoneNode):
            continue
        edge_data: dict[str, Any] = {"source": f"node{parent.id}", "target": f"node{child.id}"}
        classes = []
        if (parent, child) in highlight_edges:
            classes.append("highlight")
        if (parent, child) in combined_lin_edges:
            classes.append("combine_lin")
        elements.append({"data": edge_data, "classes": " ".join(classes)})

    return elements
