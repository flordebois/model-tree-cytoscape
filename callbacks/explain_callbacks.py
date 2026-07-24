"""Callback for the Explain page: compute + render SHAP-like plots. Unchanged placeholder."""
from dash import Input, Output, State

import ids


def register_callbacks(app):

    @app.callback(
        Output(ids.EXPLAIN_OUTPUT_CONTAINER, "children"),
        Input(ids.EXPLAIN_COMPUTE_BUTTON, "n_clicks"),
        State(ids.EXPLAIN_PLOT_TYPE_DROPDOWN, "value"),
        State(ids.STORE_VIZ_TREE, "data"),
        prevent_initial_call=True,
    )
    def compute_explanation(n_clicks, plot_type, viz_tree_dict):
        # TODO: port your KernelExplainer + predsplots/contributions logic here
        raise NotImplementedError
