# print("importing PILOT...")
# from pilot.pilot import PILOT
#
# print("import done")
from m5py import M5Prime
import os
import pickle
import time
from dash import Input, Output, State, ctx
from dash.exceptions import PreventUpdate
import numpy as np
from pmlb import fetch_data

import ids
from config import DIR_SAVED_VIZ_TREES
import dash_bootstrap_components as dbc
from dash import html
from viz_tree.viz_tree import VizTree
from viz_tree.build_viz_tree_pilot import build_viz_tree_from_pilot
from viz_tree.build_viz_tree_m5 import build_viz_tree_from_m5
from benchmark_info import PMLB_DATASETS_CAT_IDS

def format_tree_info(
        dataset_name: str,
        method_name: str,
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
        badges.append(dbc.Badge("No Collapsed Nodes", color="warning", className="me-2"))
    else:
        badges.append(dbc.Badge(f"{collapsed_nodes_count} Collapsed Nodes", color="warning", className="me-2"))
    if highlight_x is not None:
        badges.append(dbc.Badge(f"Highlighted point", color="danger", className="me-2"))
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
            ], width=6),
            dbc.Col([
                html.Small("Method", className="text-muted d-block fw-bold"),
                html.Span(children=method_name, className="fs-5")
            ], width=6),
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

def fit_new_tree(dataset_name, method_name, max_depth, max_model_depth, min_sample_split, min_sample_leaf) -> VizTree:
    print('Fitting model dataset:')
    print(dataset_name, max_depth, max_model_depth, min_sample_split, min_sample_leaf)
    data = fetch_data(dataset_name)
    X = np.array(data.iloc[:, :-1])
    y = np.array(data.iloc[:, -1])
    if method_name == "Pilot":
        categorical_ids = PMLB_DATASETS_CAT_IDS[dataset_name]
        model = PILOT(max_depth=max_depth,
                            max_model_depth=max_depth,
                            min_sample_split=min_sample_split,
                            min_sample_leaf=min_sample_leaf,
                            )
        model.fit(X, y, categorical=categorical_ids)
        pilot_tree = model.model_tree
        root_node = build_viz_tree_from_pilot(
            pilot_node=pilot_tree,
            X_train=X,
            current_indices=np.ones(len(X), dtype=bool),
            current_y_res=y,
            accumulated_coefficients=np.zeros(X.shape[1]),
            accumulated_intercept=0.0
        )

    elif method_name == "M5":
        model = M5Prime(
            use_pruning=True,
            use_smoothing=True,
            min_samples_leaf=min_sample_leaf,
            min_samples_split=min_sample_split,
            random_state=42,
            max_depth=max_depth,
        )
        model.fit(X, y)
        root_node = build_viz_tree_from_m5(model, X, y)
    else:
        raise ValueError(f'Method name {method_name} not recognized.')
    return VizTree(root_node, X, y, model.predict(X))

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
        Output(ids.ELEMENTS_TRIGGER, "data", allow_duplicate=True),

        Input(ids.BTN_LOAD_TREE, "n_clicks"),
        State(ids.INPUT_LOAD_TREE, "value"),
        prevent_initial_call=True,
    )
    def load_tree(n_clicks, load_path):
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if triggered_id != ids.BTN_LOAD_TREE:
            print(f"call to load tree with id:{triggered_id}")
            raise PreventUpdate##
        if n_clicks is None:
            print("call to load tree with clicks None")
            raise PreventUpdate#
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
        Output(ids.ELEMENTS_TRIGGER, "data", allow_duplicate=True),

        Input(ids.BTN_RELOAD_TREE, "n_clicks"),
        State(ids.STORE_VIZ_TREE_BASE, "data"),
        State(ids.STORE_TREE_PARAMS_BASE, "data"),
        prevent_initial_call=True,
    )
    def reload_tree(n_clicks, viz_tree_dict, tree_params):
        if n_clicks is None:
            print("call to reload tree with clicks None")
            raise PreventUpdate#
        return  viz_tree_dict, tree_params, time.time()

    # --- FIT NEW TREE ---
    @app.callback(
        Output(ids.STORE_VIZ_TREE, "data", allow_duplicate=True),
        Output(ids.STORE_VIZ_TREE_BASE, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS, "data", allow_duplicate=True),
        Output(ids.STORE_TREE_PARAMS_BASE, "data", allow_duplicate=True),
        Output(ids.ELEMENTS_TRIGGER, "data", allow_duplicate=True),

        Input(ids.BTN_FIT_NEW_TREE, "n_clicks"),
        State(ids.INPUT_DATASET, "value"),
        State(ids.INPUT_METHOD, "value"),
        State(ids.INPUT_MAX_DEPTH, "value"),
        State(ids.INPUT_MAX_MODEL_DEPTH, "value"),
        State(ids.INPUT_MIN_SAMPLE_SPLIT, "value"),
        State(ids.INPUT_MIN_SAMPLE_LEAF, "value"),
        prevent_initial_call=True,
    )
    def fit_tree(n_clicks, dataset_name, method_name, max_depth, max_model_depth, min_sample_split, min_sample_leaf):
        if n_clicks is None:
            print("call to fit tree with clicks None")
            raise PreventUpdate#
        print("fitting tree")
        start_time = time.time()
        viz_tree = fit_new_tree(dataset_name, method_name, max_depth, max_model_depth, min_sample_split, min_sample_leaf)
        elapsed_time = time.time() - start_time
        viz_tree_dict = viz_tree.to_dict()
        tree_params = {
            "dataset_name": dataset_name,
            "method_name": method_name,
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
        output_path = str(DIR_SAVED_VIZ_TREES / f"tree_{viz_tree_dict['tree_id']}__{tree_params['dataset_name']}-{tree_params['method_name']}-{tree_params['max_depth']}-{tree_params['max_model_depth']}-{tree_params['min_sample_split']}-{tree_params['min_sample_leaf']}.pkl")
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
        file_name = f"tree_{viz_tree_dict['tree_id']}__{tree_params['dataset_name']}-{tree_params['method_name']}-{tree_params['max_depth']}-{tree_params['max_model_depth']}-{tree_params['min_sample_split']}-{tree_params['min_sample_leaf']}"
        return {"type": "svg", "action": "download", "filename": file_name}
