"""
SHAP-like explanation view: predsplots, linear/split/global contributions.
Unchanged from the previous skeleton -- kept as its own page since
computing these is slow (KernelExplainer-based) and the controls here are
a different task from browsing/styling the tree.
"""
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

import ids
from components.loading import with_spinner

dash.register_page(__name__, path="/explain", name="Explain")

layout = dbc.Row(
    [
        dbc.Col(
            [
                dbc.Button(
                    "Compute SHAP-like explanation",
                    id=ids.EXPLAIN_COMPUTE_BUTTON,
                    class_name="mb-3 w-100",
                ),
                dcc.Dropdown(
                    id=ids.EXPLAIN_PLOT_TYPE_DROPDOWN,
                    options=[
                        "predsplot",
                        "linear contribution",
                        "split contribution",
                        "global contribution",
                    ],
                    placeholder="Plot type",
                ),
            ],
            width=3,
        ),
        dbc.Col(
            with_spinner(html.Div(id=ids.EXPLAIN_OUTPUT_CONTAINER), spinner_id="explain-spinner"),
            width=9,
        ),
    ]
) #TODO: look at explain page
