# print("importing PILOT...")
# from pilot.pilot import PILOT
#
# print("import done")
import numpy as np
from pmlb import fetch_data

import dash_cytoscape as cyto

cyto.load_extra_layouts()

from viz_tree.viz_tree import VizTree
from viz_tree.build_viz_tree import build_viz_tree_from_pilot
from benchmark_info import PMLB_DATASETS_CAT_IDS
import dash_bootstrap_components as dbc
from dash import html

def fit_new_tree(dataset_name, max_depth, max_model_depth, min_sample_split, min_sample_leaf) -> VizTree:
    print('Fitting model dataset:')
    print(dataset_name, max_depth, max_model_depth, min_sample_split, min_sample_leaf)
    data = fetch_data(dataset_name)
    X = np.array(data.iloc[:, :-1])
    y = np.array(data.iloc[:, -1])
    categorical_ids = PMLB_DATASETS_CAT_IDS[dataset_name]
    pilot_model = PILOT(max_depth=max_depth,
                        max_model_depth=max_depth,
                        min_sample_split=min_sample_split,
                        min_sample_leaf=min_sample_leaf,
                        )
    pilot_model.fit(X, y, categorical=categorical_ids)
    pilot_tree = pilot_model.model_tree
    root_node = build_viz_tree_from_pilot(
        pilot_node=pilot_tree,
        X_train=X,
        current_indices=np.ones(len(X), dtype=bool),
        current_y_res=y,
        accumulated_coefficients=np.zeros(X.shape[1]),
        accumulated_intercept=0.0
    )
    viz_tree = VizTree(root_node, X, y, pilot_model.predict(X))
    return viz_tree


