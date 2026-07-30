import time
from dash import Input, Output, State

import ids
_RANK_DIR_TO_DAGRE = {"Top Bottom": "TB", "Left Right": "LR"}

def register_callbacks(app):
    @app.callback(
        Output(ids.LAYOUT_COLLAPSE, "is_open"),
        Input(ids.LAYOUT_TOGGLE_BUTTON, "n_clicks"),
        State(ids.LAYOUT_COLLAPSE, "is_open"),
        prevent_initial_call=True,
    )
    def toggle_card(n_clicks, is_open):
        return not is_open

    @app.callback(
        Output(ids.CYTOSCAPE_GRAPH, "layout", allow_duplicate=True),
        Output(ids.DEBUG_INFO, "children", allow_duplicate=True),

        Input(ids.BTN_LAYOUT, "n_clicks"),
        Input(ids.RANK_DIR, "value"),
        Input(ids.RANK_SEP, "value"),
        Input(ids.NODE_SEP, "value"),
        prevent_initial_call=True,
    )
    def update_layout(n_clicks, rank_dir, rank_sep, node_sep):
        return {
            "name": "dagre",
            "rankDir": _RANK_DIR_TO_DAGRE[rank_dir],
            "rankSep": rank_sep,
            "nodeSep": node_sep,
            "animate": True,
            "fit": True,
            "_time": time.time(),  # dummy changing field, forces cytoscape to re-layout
        }, "Layout updated."
