from m5py import M5Prime
import os
import pickle
import time
from dash import Input, Output, State, ctx, callback, no_update
from dash.exceptions import PreventUpdate
import numpy as np
import base64
from pathlib import Path

import ids
from config import DIR_SAVED_VIZ_TREES, NO_FILE_SELECTED_PLACEHOLDER, DEFAULT_DATASET_NAME
from viz_tree.viz_tree import VizTree
from viz_tree.build_viz_tree_pilot import build_viz_tree_from_pilot
from viz_tree.build_viz_tree_m5 import build_viz_tree_from_m5
from benchmark_info import PMLB_DATASETS_CAT_IDS

from dataset.dataset_registry import (
    NEW_CSV_OPTION,
    DIR_DATASET_UPLOAD,
    build_dropdown_options,
    get_target_col,
)
from dataset.dataset import Dataset

def fit_new_tree(input_dataset, method_name, max_depth, max_model_depth, min_sample_split, min_sample_leaf) -> (VizTree, str):
    print('Fitting model dataset:')
    print(input_dataset, max_depth, max_model_depth, min_sample_split, min_sample_leaf)

    if input_dataset.endswith(".csv"):
        csv_path = Path(input_dataset)
        dataset_name = input_dataset[:-4]
        dataset = Dataset.from_csv(input_dataset, target_col=get_target_col(csv_path), name=dataset_name)
    elif input_dataset.endswith(".pmlb"):
        dataset_name = input_dataset[:-5]
        dataset = Dataset.from_pmlb(dataset_name)
    else:
        raise ValueError("Unknown dataset type")

    X = dataset.X
    y = dataset.y
    if method_name == "Pilot":
        print("importing PILOT...")
        from pilot.pilot import PILOT
        print("import done")
        start_time = time.time()
        model = PILOT(max_depth=max_depth,
                      max_model_depth=max_depth,
                      min_sample_split=min_sample_split,
                      min_sample_leaf=min_sample_leaf,
                      )
        model.fit(X, y, categorical=dataset.cat_ids)
        pilot_tree = model.model_tree
        root_node = build_viz_tree_from_pilot(
            pilot_node=pilot_tree,
            X_train=X,
            current_indices=np.ones(len(X), dtype=bool),
            current_y_res=y,
            accumulated_coefficients=np.zeros(X.shape[1]),
            accumulated_intercept=0.0
        )
        elapsed_time = time.time() - start_time
    elif method_name == "M5":
        start_time = time.time()
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
        elapsed_time = time.time() - start_time
    else:
        raise ValueError(f'Method name {method_name} not recognized.')
    return VizTree(root_node, X, y, model.predict(X)), f"{int(elapsed_time // 60)}min {int(elapsed_time % 60)}sec"

def register_callbacks(app):
    @app.callback(
        Output(ids.NEW_TREE_COLLAPSE, "is_open"),
        Input(ids.NEW_TREE_TOGGLE_BUTTON, "n_clicks"),
        State(ids.NEW_TREE_COLLAPSE, "is_open"),
        prevent_initial_call=True,
    )
    def toggle_card(n_clicks, is_open):
        return not is_open

    # --- OPEN CSV MODAL ---
    @app.callback(
        Output(ids.MODAL_CSV, "is_open"),
        Input(ids.INPUT_DATASET, "value"),
        prevent_initial_call=True,
    )
    def maybe_open_csv_modal(value):
        if value == NEW_CSV_OPTION:
            return True
        raise PreventUpdate

    # --- CANCEL CSV MODAL ---
    @app.callback(
        Output(ids.MODAL_CSV_FEEDBACK, "children", allow_duplicate=True),
        Output(ids.MODAL_CSV, "is_open", allow_duplicate=True),
        Output(ids.INPUT_DATASET, "value", allow_duplicate=True),
        Output(ids.MODAL_CSV_FILE_NAME, "children", allow_duplicate=True),
        Output(ids.MODAL_CSV_INPUT_TARGET_COL, "value", allow_duplicate=True),

        Input(ids.MODAL_CSV_BTN_CANCEL, "n_clicks"),
        State(ids.MODAL_CSV_FILE_NAME, "children"),
        prevent_initial_call=True,
    )
    def cancel_csv_modal(_, file_name):
        file_path = str(DIR_DATASET_UPLOAD / file_name)
        if file_path != NO_FILE_SELECTED_PLACEHOLDER:
            path = Path(file_path)
            if path.exists():
                path.unlink()
        return None, False, DEFAULT_DATASET_NAME, NO_FILE_SELECTED_PLACEHOLDER, None

    # --- CONFIRM CSV MODAL ---
    @app.callback(
        Output(ids.MODAL_CSV_FEEDBACK, "children", allow_duplicate=True),
        Output(ids.MODAL_CSV, "is_open", allow_duplicate=True),
        Output(ids.INPUT_DATASET, "options", allow_duplicate=True),
        Output(ids.INPUT_DATASET, "value", allow_duplicate=True),
        Output(ids.MODAL_CSV_FILE_NAME, "children", allow_duplicate=True),
        Output(ids.MODAL_CSV_INPUT_TARGET_COL, "value", allow_duplicate=True),

        Input(ids.MODAL_CSV_BTN_CONFIRM, "n_clicks"),
        State(ids.MODAL_CSV_FILE_NAME, "children"),
        State(ids.MODAL_CSV_INPUT_TARGET_COL, "value"),
        prevent_initial_call=True,
    )
    def confirm_csv_modal(_, file_name, target_col):
        file_path = str(DIR_DATASET_UPLOAD / file_name)
        if file_path == NO_FILE_SELECTED_PLACEHOLDER:
            return "No file was uploaded yet.", no_update, no_update, no_update, no_update, no_update

        if not target_col:
            return "No target column was given.", no_update, no_update, no_update, no_update, no_update

        try:
            Dataset.from_csv(file_path, target_col=target_col)
        except (ValueError, KeyError, TypeError) as e:
            return f"Could not load CSV: {e}", no_update, no_update, no_update, no_update, no_update

        path = Path(file_path)
        path.with_suffix(".target.txt").write_text(target_col)
        return None, False, build_dropdown_options(), str(path), NO_FILE_SELECTED_PLACEHOLDER, None

    # --- CSV UPLOAD ---
    @app.callback(
        Output(ids.MODAL_CSV_FILE_NAME, "children"),
        Output(ids.MODAL_CSV_FEEDBACK, "children"),

        Input(ids.MODAL_CSV_UPLOAD, "contents"),
        State(ids.MODAL_CSV_UPLOAD, "filename"),
        State(ids.MODAL_CSV_FILE_NAME, "children"),
        prevent_initial_call=True,
    )
    def csv_upload(contents, filename, old_file_name):
        DIR_DATASET_UPLOAD.mkdir(parents=True, exist_ok=True)

        old_file_path = str(DIR_DATASET_UPLOAD / old_file_name)
        if old_file_path != NO_FILE_SELECTED_PLACEHOLDER:
            path = Path(old_file_path)
            if path.exists():
                path.unlink()

        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)

        path = DIR_DATASET_UPLOAD / filename

        if path.exists():
            return no_update, f"There already exists a dataset with the name {filename}."

        path.write_bytes(decoded)
        return filename, no_update


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
        Output(ids.DUMMY_FOR_SPINNER, "children"),

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
        viz_tree, training_time = fit_new_tree(dataset_name, method_name, max_depth, max_model_depth, min_sample_split, min_sample_leaf)
        viz_tree_dict = viz_tree.to_dict()
        tree_params = {
            "dataset_name": dataset_name,
            "method_name": method_name,
            "max_depth": max_depth,
            "max_model_depth": max_model_depth,
            "min_sample_split": min_sample_split,
            "min_sample_leaf": min_sample_leaf,
            "training_time": training_time,
            "subtree_node_id": -1,
            "collapsed_nodes_count": 0,
            "highlight_x": None
        }

        return viz_tree_dict, viz_tree_dict, tree_params, tree_params, time.time(), ""

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
