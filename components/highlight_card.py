"""Highlight card: highlight a specific point's prediction path through the tree."""
import dash_bootstrap_components as dbc
from dash import dcc, html

import ids


def make_highlight_card() -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader("Highlight"),
            dbc.CardBody(
                [
                    html.Div(
                        style={"display": "flex", "gap": "8px", "marginBottom": "10px", "flexWrap": "wrap"},
                        children=[
                            dbc.Button("Highlight", id=ids.BTN_HIGHLIGHT, size="sm"),
                            dbc.Button("Clear highlight", id=ids.BTN_CLEAR_HIGHLIGHT, size="sm"),
                            dbc.Button("Random point", id=ids.BTN_RANDOM_POINT, size="sm"),
                        ],
                    ),
                    dbc.Input(
                        id=ids.INPUT_HIGHLIGHT,
                        type="text",
                        placeholder="e.g. 1.2, 0.5, 3.1",
                        class_name="mb-2",
                    ),
                    dcc.Checklist(
                        id=ids.HIGHLIGHT_OPTIONS,
                        options=[{"label": "Only show highlighted path", "value": "only_show_highlight"}],
                        value=[],
                        inline=True,
                    ),
                ]
            ),
        ],
        class_name="mb-3",
    )
