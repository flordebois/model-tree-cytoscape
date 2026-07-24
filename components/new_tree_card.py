"""
New Tree card: fit or load a tree, plus save/download/reload -- everything
about the tree's lifecycle. Controls and defaults copied from the old
"Tree" card's dataset/fit section.
"""
import dash_bootstrap_components as dbc
from dash import dcc, html

import ids
from components.loading import with_spinner
from config import (
    DATASET_OPTIONS,
    DEFAULT_DATASET_NAME,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_MODEL_DEPTH,
    DEFAULT_MIN_SAMPLE_LEAF,
    DEFAULT_MIN_SAMPLE_SPLIT,
)


def make_new_tree_card() -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader("New Tree"),
            dbc.CardBody(
                [
                    html.Div(
                        style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "10px"},
                        children=[
                            html.Div(
                                [
                                    html.Div("Dataset", style={"marginBottom": "4px"}),
                                    dcc.Dropdown(
                                        id=ids.INPUT_DATASET,
                                        options=DATASET_OPTIONS,
                                        value=DEFAULT_DATASET_NAME,
                                        clearable=False,
                                        style={"width": "220px"},
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Div("Max depth", style={"marginBottom": "4px"}),
                                    dcc.Input(id=ids.INPUT_MAX_DEPTH, type="number", value=DEFAULT_MAX_DEPTH),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Div("Max model depth", style={"marginBottom": "4px"}),
                                    dcc.Input(
                                        id=ids.INPUT_MAX_MODEL_DEPTH, type="number", value=DEFAULT_MAX_MODEL_DEPTH
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Div("Min sample split", style={"marginBottom": "4px"}),
                                    dcc.Input(
                                        id=ids.INPUT_MIN_SAMPLE_SPLIT, type="number", value=DEFAULT_MIN_SAMPLE_SPLIT
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Div("Min sample leaf", style={"marginBottom": "4px"}),
                                    dcc.Input(
                                        id=ids.INPUT_MIN_SAMPLE_LEAF, type="number", value=DEFAULT_MIN_SAMPLE_LEAF
                                    ),
                                ]
                            ),
                        ],
                    ),
                    html.Div(
                        style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "10px"},
                        children=[
                            dbc.Button("Fit PILOT tree", id=ids.BTN_FIT_NEW_TREE, size="sm"),
                            dbc.Button("Save shown tree", id=ids.BTN_SAVE_TREE, size="sm"),
                            dbc.Button("Download shown tree", id=ids.BTN_SAVE_TREE_SVG, size="sm"),
                            dbc.Button("Reload tree", id=ids.BTN_RELOAD_TREE, size="sm"),
                        ],
                    ),
                    html.Div(
                        style={"display": "flex", "gap": "8px"},
                        children=[
                            dbc.Input(
                                id=ids.INPUT_LOAD_TREE,
                                type="text",
                                placeholder="path to saved_viz_trees/...",
                            ),
                            dbc.Button("Load tree", id=ids.BTN_LOAD_TREE, size="sm"),
                        ],
                    ),
                    # Fitting is the slow action here -> spinner-wrapped feedback area.
                    with_spinner(html.Div(id=ids.NEW_TREE_FIT_OUTPUT), spinner_id=ids.NEW_TREE_FIT_SPINNER),
                ]
            ),
        ],
        class_name="mb-3",
    )
