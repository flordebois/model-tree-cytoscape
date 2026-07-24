import time
import numpy as np
from dash import Input, Output, State, ctx
from dash.exceptions import PreventUpdate

import ids

def register_callbacks(app):
    @app.callback(
        Output(ids.INPUT_HIGHLIGHT, "value"),
        Input(ids.BTN_RANDOM_POINT, "n_clicks"),
        State(ids.STORE_VIZ_TREE, "data"),
        prevent_initial_call=True,
    )
    def set_random_point(n_clicks, viz_tree_dict):
        X_train = np.array(viz_tree_dict["X_train"])
        return ", ".join(map(str, X_train[np.random.randint(X_train.shape[0])]))

    @app.callback(
        Output(ids.STORE_HIGHLIGHT_X, "data", allow_duplicate=True),
        Output(ids.DEBUG_INFO, "children", allow_duplicate=True),
        Output(ids.ELEMENTS_TRIGGER, "data", allow_duplicate=True),

        Input(ids.BTN_HIGHLIGHT, "n_clicks"),
        State(ids.INPUT_HIGHLIGHT, "value"),
        State(ids.STORE_VIZ_TREE, "data"),
        prevent_initial_call=True,
    )
    def highlight_path(n_clicks, input_highlight_x, viz_tree_dict):
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if triggered_id != ids.BTN_HIGHLIGHT or n_clicks is None or input_highlight_x is None:
            raise PreventUpdate
        n_features = np.array(viz_tree_dict["X_train"]).shape[1]
        try:
            highlight_x = np.array([float(v.strip()) for v in input_highlight_x.split(",")])
        except ValueError:
            return None, "Couldn't parse the highlight input -- expected comma-separated numbers.", time.time()

        if len(highlight_x) != n_features:
            return (
                None,
                f"Highlight input has length {len(highlight_x)}, expected {n_features}.",
                time.time(),
            )

        return highlight_x.tolist(), f"Highlighting path of {highlight_x}.", time.time()
    
    @app.callback(
        Output(ids.STORE_HIGHLIGHT_X, "data", allow_duplicate=True),
        Output(ids.DEBUG_INFO, "children", allow_duplicate=True),
        Output(ids.ELEMENTS_TRIGGER, "data", allow_duplicate=True),

        Input(ids.BTN_CLEAR_HIGHLIGHT, "n_clicks"),
        State(ids.STORE_HIGHLIGHT_X, "data"),
        prevent_initial_call=True,
    )
    def clear_highlight(n_clicks, highlight_x):
        if highlight_x is None:
            raise PreventUpdate
        return None, "Highlight cleared.", time.time()