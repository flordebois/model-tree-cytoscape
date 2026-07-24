"""
New Tree card: fit or load a tree, plus save/download/reload -- everything
about the tree's lifecycle. Controls and defaults copied from the old
"Tree" card's dataset/fit section.
"""
import dash_bootstrap_components as dbc
from dash import dcc, html
import dash_daq as daq

import ids
from components.loading import with_spinner
from config import DEFAULT_COLLAPSE_LEVEL


def make_edit_tree_card() -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader("Edit Tree"),
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.Div("Tree structure", className="fw-bold mb-2"),
                            html.Div(
                                style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "10px"},
                                children=[
                                    dbc.Button("Collapse/expand selected node", id=ids.BTN_COLLAPSE_EXPAND, size="sm"),
                                    dbc.Button("Subtree from selected node", id=ids.BTN_SUBTREE, size="sm"),
                                    dbc.Button("Expand all nodes", id=ids.BTN_EXPAND_ALL, size="sm"),
                                ],
                            ),
                            html.Div(
                                style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap"},
                                children=[
                                    dbc.Button("Collapse nodes to level:", id=ids.BTN_COLLAPSE_LEVEL, size="sm"),
                                    dcc.Input(
                                        id=ids.INPUT_COLLAPSE_LEVEL, type="number", value=DEFAULT_COLLAPSE_LEVEL,
                                        style={"width": "80px"}
                                    ),
                                    dcc.Checklist(
                                        id=ids.COLLAPSE_LEVEL_OPTIONS,
                                        options=[{"label": "include linear nodes", "value": "include_lin"}],
                                        value=["include_lin"],
                                        inline=True,
                                    ),
                                ],
                            ),
                        ],
                        className="mb-3",
                    ),

                    html.Div(
                        [
                            html.Div("Display switches", className="fw-bold mb-2"),
                            html.Div(
                                [
                                    html.Span("Linear nodes: "),
                                    dbc.RadioItems(
                                        id=ids.SWITCH_COMBINE_LIN,
                                        options=[
                                            {"label": "Show all", "value": 0},
                                            {"label": "Combine into node", "value": 1},
                                            {"label": "Combine into edge", "value": 2},
                                        ],
                                        value=0,
                                        inline=True,
                                    ),
                                ],
                                className="mb-2",
                            ),
                            html.Div(
                                style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                                children=[
                                    daq.BooleanSwitch(id=ids.SWITCH_REG_PLOTS, label="Show regression plots"),
                                    daq.BooleanSwitch(id=ids.SWITCH_PREDS_PLOTS, label="Show prediction plots"),
                                    daq.BooleanSwitch(id=ids.SWITCH_MINIMAL, label="Minimal nodes"),
                                    daq.BooleanSwitch(id=ids.SWITCH_RSS, label="Show RSS"),
                                ],
                            ),
                        ],
                        className="mb-3",
                    )
                ]
            ),
        ],
        class_name="mb-3",
    )
