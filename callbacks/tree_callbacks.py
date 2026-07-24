import os
import pickle
import time
from dash import Input, Output, State, ctx
from dash.exceptions import PreventUpdate

import ids
from config import DIR_SAVED_VIZ_TREES
import dash_bootstrap_components as dbc
from dash import html
from placeholder import fit_new_tree

def format_tree_info(
        dataset_name: str,
        max_depth: int,
        max_model_depth: int,
        min_sample_split: int,
        min_sample_leaf: int,
        training_time: float,
        subtree_node_id: int = -1,
        collapsed_nodes_count: int = 0,
        highlight_x = None
):
    # Build status badges dynamically
    badges = []
    if subtree_node_id == -1:
        badges.append(dbc.Badge("Full tree", color="info", className="me-2"))
    else:
        badges.append(dbc.Badge(f"Subtree at node{subtree_node_id}", color="info", className="me-2"))
    if collapsed_nodes_count == 0:
        badges.append(dbc.Badge("No Collapsed Nodes", color="warning"))
    else:
        badges.append(dbc.Badge(f"{collapsed_nodes_count} Collapsed Nodes", color="warning"))
    if highlight_x is not None:
        badges.append(dbc.Badge(f"Highlighted point:{highlight_x}", color="info", className="me-2"))
    return dbc.CardBody([
        html.Div(
            className="d-flex justify-content-between align-items-center mb-3 pb-2 border-bottom",
            children=[
                html.H5("Tree Information", className="mb-0 text-primary fw-bold"),
                html.Div(badges)
            ]
        ),

        dbc.Row([
            dbc.Col([
                html.Small("Dataset name", className="text-muted d-block fw-bold"),
                html.Span(children=dataset_name, className="fs-5")
            ], width=12),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                html.Small("Max Depth (model)", className="text-muted d-block fw-bold"),
                html.Span(f"{max_depth} ({max_model_depth})", className="fs-6")
            ], width=3),
            dbc.Col([
                html.Small("Training time", className="text-muted d-block fw-bold"),
                html.Span(f"{training_time}", className="fs-6")
            ], width=3),
            dbc.Col([
                html.Small("Min sample split", className="text-muted d-block fw-bold"),
                html.Span(str(min_sample_split), className="fs-6")
            ], width=3),
            dbc.Col([
                html.Small("Min sample leaf", className="text-muted d-block fw-bold"),
                html.Span(str(min_sample_leaf), className="fs-6")
            ], width=3),
        ])
    ])

def register_callbacks(app):
    # --- UPDATE TREE PARAMS ---
    @app.callback(
        Output(ids.TREE_INFO_TEXT, "children"),

        Input(ids.STORE_TREE_PARAMS, "data"),
        prevent_initial_call=True #TODO: does this run initially or not?
    )
    def update_tree_info_card(tree_params):
        return format_tree_info(**tree_params)

    # --- LOAD TREE ---
    @app.callback(
        Output(ids.STORE_VIZ_TREE, "data", allow_duplicate=True),
        Output(ids.STORE_VIZ_TREE_BASE, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS_BASE, "data", allow_duplicate=True),
        # Output(ids.SWITCH_MINIMAL, "on", allow_duplicate=True), # TODO: Don't do it here but where viz_tree need to get loaded
        Output(ids.ELEMENTS_TRIGGER, "data", allow_duplicate=True),  # TODO: Different way?

        Input(ids.BTN_LOAD_TREE, "n_clicks"),
        State(ids.INPUT_LOAD_TREE, "value"),
        prevent_initial_call=True,
    )
    def load_tree(n_clicks, load_path):
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if triggered_id != ids.BTN_LOAD_TREE or n_clicks is None:
            raise PreventUpdate
        resolved_path = load_path
        if not resolved_path or not os.path.exists(resolved_path):
            resolved_path = max(
                (str(DIR_SAVED_VIZ_TREES / f) for f in os.listdir(str(DIR_SAVED_VIZ_TREES)) if f.endswith(".pkl")),
                key=os.path.getctime,
            )
        with open(resolved_path, "rb") as f:
            input_dict = pickle.load(f)
        
        tree_params = input_dict["tree_params"]
        viz_tree_dict = input_dict["viz_tree_dict"]
        return viz_tree_dict, viz_tree_dict, tree_params, tree_params, time.time()

    # --- RELOAD BASE TREE ---
    @app.callback(
        Output(ids.STORE_VIZ_TREE, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS, "data", allow_duplicate=True),
        # Output(ids.SWITCH_MINIMAL, "on", allow_duplicate=True), # TODO: Don't do it here but where viz_tree need to get loaded
        Output(ids.ELEMENTS_TRIGGER, "data", allow_duplicate=True),  # TODO: Different way?

        Input(ids.BTN_RELOAD_TREE, "n_clicks"),
        State(ids.STORE_VIZ_TREE_BASE, "data"),
        State(ids.STORE_TREE_PARAMS_BASE, "data"),
        prevent_initial_call=True,
    )
    def reload_tree(n_clicks, viz_tree_dict, tree_params):
        if n_clicks is None:
            raise PreventUpdate
        return  viz_tree_dict, tree_params, time.time()

    # --- FIT NEW TREE ---
    @app.callback(
        Output(ids.STORE_VIZ_TREE, "data", allow_duplicate=True),
        Output(ids.STORE_VIZ_TREE_BASE, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS_BASE, "data", allow_duplicate=True),
        # Output(ids.SWITCH_MINIMAL, "on", allow_duplicate=True), # TODO: Don't do it here but where viz_tree need to get loaded
        Output(ids.ELEMENTS_TRIGGER, "data", allow_duplicate=True),  # TODO: Different way?

        Input(ids.BTN_FIT_NEW_TREE, "n_clicks"),
        State(ids.INPUT_DATASET, "value"),
        State(ids.INPUT_MAX_DEPTH, "value"),
        State(ids.INPUT_MAX_MODEL_DEPTH, "value"),
        State(ids.INPUT_MIN_SAMPLE_SPLIT, "value"),
        State(ids.INPUT_MIN_SAMPLE_LEAF, "value"),
        prevent_initial_call=True,
    )
    def fit_tree(n_clicks, dataset_name, max_depth, max_model_depth, min_sample_split, min_sample_leaf):
        if n_clicks is None:
            raise PreventUpdate
        print("fitting tree")
        start_time = time.time()
        viz_tree = fit_new_tree(dataset_name, max_depth, max_model_depth, min_sample_split, min_sample_leaf)
        elapsed_time = time.time() - start_time
        viz_tree_dict = viz_tree.to_dict()
        tree_params = {
            "dataset_name": dataset_name,
            "max_depth": max_depth,
            "max_model_depth": max_model_depth,
            "min_sample_split": min_sample_split,
            "min_sample_leaf": min_sample_leaf,
            "training_time": f"{int(elapsed_time // 60)}min {int(elapsed_time % 60)}sec",
            "subtree_node_id": -1,
            "collapsed_nodes_count": 0,
            "highlight_x": None
        }

        return viz_tree_dict, viz_tree_dict, tree_params, tree_params, time.time()

    # --- SAVE TREE ---
    @app.callback(
        Output(ids.DEBUG_INFO, "children", allow_duplicate=True),

        Input(ids.BTN_SAVE_TREE, "n_clicks"),
        State(ids.STORE_VIZ_TREE, "data"),
        State(ids.STORE_TREE_PARAMS, "data"),
        prevent_initial_call=True,
    )
    def save_tree(n_clicks, viz_tree_dict, tree_params):
        print("saving tree")
        output_path = str(DIR_SAVED_VIZ_TREES / f"tree_{viz_tree_dict['tree_id']}.pkl")
        output_dict = {"tree_params": tree_params, 
                       "viz_tree_dict": viz_tree_dict}
        with open(output_path, "wb") as handle:
            # noinspection PyTypeChecker
            pickle.dump(output_dict, handle)
        return f"Tree saved to {output_path}"

    # --- DOWNLOAD SVG ---
    @app.callback(
        Output(ids.CYTOSCAPE_GRAPH, "generateImage"),

        Input(ids.BTN_SAVE_TREE_SVG, "n_clicks"),
        State(ids.STORE_VIZ_TREE, "data"),
        State(ids.STORE_TREE_PARAMS, "data"),
        prevent_initial_call=True,
    )
    def download_tree_svg(n_clicks, viz_tree_dict, tree_params):
        print("downloading tree")
        file_name = f"tree{viz_tree_dict['tree_id']}__{tree_params['dataset_name']}-{tree_params['max_depth']}-{tree_params['max_model_depth']}-{tree_params['min_sample_split']}-{tree_params['min_sample_leaf']}"
        return {"type": "svg", "action": "download", "filename": file_name}
