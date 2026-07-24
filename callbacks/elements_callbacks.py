import time
import numpy as np
from dash import Input, Output, State
from dash.exceptions import PreventUpdate

import ids
from config import DIR_LIVE_OUTPUT
from viz_tree.viz_tree import VizTree
from viz_tree.viz_tree_cytoscape import viz_tree_to_cytoscape_elements


def register_callbacks(app):
    @app.callback(
        Output(ids.ELEMENTS_TRIGGER, "data", allow_duplicate=True),

        Input(ids.SWITCH_COMBINE_LIN, "value"),
        # Input(ids.SWITCH_REG_PLOTS, "on"), #TODO: fix
        # Input(ids.SWITCH_PREDS_PLOTS, "on"),
        Input(ids.SWITCH_RSS, "on"),
        Input(ids.SWITCH_MINIMAL, "on"),
        prevent_initial_call=True,
    )
    def bump_trigger(*_):
        return time.time()

    @app.callback(
        Output(ids.CYTOSCAPE_GRAPH, "elements", allow_duplicate=True),

        Input(ids.ELEMENTS_TRIGGER, "data"),
        State(ids.SWITCH_COMBINE_LIN, "value"),
        State(ids.SWITCH_REG_PLOTS, "on"),
        State(ids.SWITCH_PREDS_PLOTS, "on"),
        State(ids.SWITCH_MINIMAL, "on"),
        State(ids.SWITCH_RSS, "on"),
        State(ids.STORE_VIZ_TREE, "data"),
        State(ids.INPUT_DISPLAY_TYPE, "value"),
        State(ids.INPUT_NMAX, "value"),
        State(ids.INPUT_FIG_W, "value"),
        State(ids.INPUT_FIG_H, "value"),
        State(ids.PREDSPLOT_OPTIONS, "value"),
        State(ids.STORE_HIGHLIGHT_X, "data"),
        State(ids.HIGHLIGHT_OPTIONS, "value"),
        State(ids.PREDSPLOT_TYPE, "value"),
        prevent_initial_call=True,
    )
    def update_elements(
        trigger,
        combine_lin,
        use_regplots,
        use_predsplots,
        use_minimal,
        show_rss,
        viz_tree_dict,
        display_type,
        nmax,
        figw,
        figh,
        predsplot_options,
        highlight_x,
        highlight_options,
        predsplot_type,
    ):
        if viz_tree_dict is None:
            print("trigger without viz_tree_dict")
            raise PreventUpdate

        use_intercept = "intercept" in predsplot_options
        truncate_total_pred = "truncate" in predsplot_options
        staircase = "staircase" in predsplot_options
        type2 = "type2" in predsplot_type
        only_show_highlight = "only_show_highlight" in highlight_options

        viz_tree = VizTree.from_dict(viz_tree_dict)
        highlight_x_arr = None if highlight_x is None else np.array(highlight_x)

        elements = viz_tree_to_cytoscape_elements(
            viz_tree,
            str(DIR_LIVE_OUTPUT),
            combine_lin=combine_lin,
            show_rss=show_rss,
            use_regplots=use_regplots,
            use_predsplots=use_predsplots,
            fig_size=(figw, figh),
            predsplot_n_max=nmax,
            predsplot_use_intercept=use_intercept,
            predsplot_display_type=display_type,
            predsplot_truncate_total_pred=truncate_total_pred,
            predsplot_staircase=staircase,
            predsplot_type2=type2,
            highlight_x=highlight_x_arr,
            only_show_highlight=only_show_highlight,
        )

        if use_minimal: #TODO: change to layout
            for el in elements:
                if "data" in el and "id" in el["data"]:  # node, not edge
                    classes = el.get("classes", "").split()
                    classes.append("minimal")
                    el["classes"] = " ".join(classes)

        return elements
