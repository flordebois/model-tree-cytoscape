import ids
from dash import Input, Output, State, html
import numpy as np
import matplotlib.pyplot as plt

from callbacks.d_edit_tree_callbacks import find_node_by_cytoscape_id
from config import DIR_LIVE_OUTPUT, NODE_TYPE_COLORS
from nodes.internal_node import InternalNode
from nodes.leaf_node import LeafNode
from plots.predsplot import predsplot
from plots.predsplot2 import predsplot2
from plots.regplot import make_regression_plot
from viz_tree.viz_tree import VizTree
from urllib.parse import quote

def register_callbacks(app):
    @app.callback(
        Output(ids.NODE_INFO_TYPE, "children"),
        Output(ids.NODE_INFO_TYPE, "style"),
        Output(ids.NODE_INFO_LABEL, "children"),
        Output(ids.NODE_INFO_ID, "children"),
        Output(ids.NODE_INFO_SAMPLES, "children"),
        Output(ids.NODE_INFO_RSS, "children"),
        Output(ids.NODE_INFO_RSS_REDUCTION, "children"),

        Input(ids.CYTOSCAPE_GRAPH, "tapNodeData"),
        prevent_initial_call=True
    )
    def update_node_info(data):
        node_type = data.get("node_type", "—") if data else "—"
        badge_label = "Standby" if not data else f"{node_type[:-4]} Node"
        badge_color = NODE_TYPE_COLORS.get(node_type, "#6c757d")
        badge_style = {"backgroundColor": badge_color, "color": "#ffffff"}

        label = data.get("label", "—") if data else "No node selected."
        node_id = data.get("id", "—") if data else "—"
        n_samples = data.get("n_samples", "—") if data else "—"
        rss = data.get("rss", "—") if data else "—"
        rss_root_reduction = data.get("rss_root_reduction", "—") if data else "—"
        return badge_label, badge_style, label, node_id, n_samples, rss, rss_root_reduction


    @app.callback(
        Output(ids.NODE_INFO_PLOT_COLLAPSE, "is_open"),
        Input(ids.NODE_INFO_PLOT_SWITCH, "value"),
        prevent_initial_call=True
    )
    def toggle_node_plot(switch_on):
        return bool(switch_on)

    @app.callback(
        Output(ids.NODE_INFO_PLOT_CONTAINER, "children"),

        Input(ids.NODE_INFO_PLOT_SWITCH, "value"),
        Input(ids.CYTOSCAPE_GRAPH, "tapNodeData"),
        Input(ids.BTN_REFIT_PLOTS, "n_clicks"),
        State(ids.STORE_VIZ_TREE, "data"),

        State(ids.INPUT_DISPLAY_TYPE, "value"),
        State(ids.INPUT_NMAX, "value"),
        State(ids.INPUT_FIG_W, "value"),
        State(ids.INPUT_FIG_H, "value"),
        State(ids.PREDSPLOT_OPTIONS, "value"),
        State(ids.STORE_HIGHLIGHT_X, "data"),
        State(ids.PREDSPLOT_TYPE, "value"),

        prevent_initial_call=True,
    )
    def render_node_plot(switch_on, tapped_node, _n_clicks, viz_tree_dict,
                         display_type,
                         nmax,
                         figw,
                         figh,
                         predsplot_options,
                         highlight_x,
                         predsplot_type,
                         ):
        if not switch_on or not tapped_node:
            return "Plot will appear here, no node selected."

        use_intercept = "intercept" in predsplot_options
        truncate_total_pred = "truncate" in predsplot_options
        staircase = "staircase" in predsplot_options
        type2 = "type2" in predsplot_type
        highlight_x_arr = None if highlight_x is None or not tapped_node['highlight'] else np.array(highlight_x)

        viz_tree = VizTree.from_dict(viz_tree_dict)
        node = find_node_by_cytoscape_id(viz_tree, tapped_node["id"])
        n_features = viz_tree.X_train.shape[1]
        cmap = plt.colormaps['tab20'].resampled(n_features)
        feature_colors = [cmap(i) for i in range(n_features)]

        if isinstance(node, LeafNode):
            if type2:
                file_dir = DIR_LIVE_OUTPUT / "predsplots" / f"predsplot2_node{node.id}_{viz_tree.tree_id}.svg"
                predsplot2(viz_tree=viz_tree,
                           leaf_node=node,
                           y_hat = viz_tree.y_hat,
                           n_max=nmax,
                           fig_size=(figw, figh),
                           truncate_total_pred=truncate_total_pred,
                           variable_tick_width=True,
                           display_type=display_type,
                           file_directory=str(file_dir),
                           highlight_x=highlight_x_arr,
                           staircase=staircase,
                           feature_names=None,
                           all_feature_colors=feature_colors,
                           )
            else:
                if np.any(np.array(node.node_model.coefficients) != 0):
                    file_dir = DIR_LIVE_OUTPUT / "predsplots" / f"predsplot_node{node.id}_{viz_tree.tree_id}.svg"
                    node_X = viz_tree.X_train[node.indices, :]
                    if use_intercept:
                        intercept = node.node_model.intercept
                    else:
                        intercept = None
                    predsplot(node_X,
                              np.array(node.node_model.coefficients),
                              y_hat=np.array(np.sum(node.node_model.coefficients * node_X, axis=1) + node.node_model.intercept),
                              n_max=nmax,
                              intercept=intercept,
                              fig_size=(figw, figh),
                              feature_names=None,
                              all_feature_colors=feature_colors,
                              display_type=display_type,
                              truncate_total_pred=truncate_total_pred,
                              variable_tick_width=True,
                              file_directory=str(file_dir),
                              highlight_x=highlight_x_arr,
                              staircase=staircase)
                else:
                    return "Prediction plots (type 1) can't be made for nodes with no linear model."
        elif isinstance(node, InternalNode):
            file_dir = DIR_LIVE_OUTPUT / "regplots" / f"regplot_node{node.id}_{viz_tree.tree_id}.svg"
            make_regression_plot(
                node,
                viz_tree.X_train,
                str(file_dir),
                (figw, figh),
                feature_colors,
                None,
                highlight_x_arr
            )
        else:
            return f"No plot available for class {node.__class__.__name__}."

        if file_dir is None:
            return (
                "Error, plot no file was created."
            )

        svg_data = file_dir.read_text(encoding="utf-8")
        encoded_svg = quote(svg_data)

        src_url =  f"data:image/svg+xml;utf8,{encoded_svg}"
        return html.Img(src=src_url, style={"maxWidth": "100%"})
