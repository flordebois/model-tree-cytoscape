"""
Layout card: purely about how the tree is laid out on screen (dagre
rank direction / spacing). Copied from the old "Controls card", minus
the display switches (those moved to the Settings card).
"""
import dash_bootstrap_components as dbc
from dash import dcc, html

import ids
from config import DEFAULT_NODE_SEP, DEFAULT_RANK_SEP


def make_layout_card() -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader("Layout"),
            dbc.CardBody(
                [
                    dbc.Button("Refit layout", id=ids.BTN_LAYOUT, size="sm", class_name="mb-3"),
                    html.Div(
                        [
                            "Display direction:",
                            dcc.RadioItems(
                                id=ids.RANK_DIR,
                                options=["Top Bottom", "Left Right"],
                                value="Top Bottom",
                                inline=True,
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.Div(
                        [
                            "rankSep (vertical gap)",
                            dcc.Slider(id=ids.RANK_SEP, min=5, max=100, step=1, value=DEFAULT_RANK_SEP, marks=None),
                        ],
                        className="mb-2",
                    ),
                    html.Div(
                        [
                            "nodeSep (horizontal gap)",
                            dcc.Slider(id=ids.NODE_SEP, min=5, max=100, step=1, value=DEFAULT_NODE_SEP, marks=None),
                        ]
                    ),
                ]
            ),
        ],
        class_name="mb-3",
    )
