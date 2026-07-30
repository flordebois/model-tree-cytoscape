import dash_bootstrap_components as dbc
from dash import html, dcc
from components.cards.bb_node_plot_settings_card import make_settings_card
import ids

def make_node_info_card() -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardBody([
                html.Div(

                    className="d-flex justify-content-between align-items-center mb-3 pb-2 border-bottom",
                    children=[
                        html.H5("Node Information", className="mb-0 text-primary fw-bold"),
                        html.Span(
                            "Standby",
                            id=ids.NODE_INFO_TYPE,
                            className="badge fs-6",
                            style={"backgroundColor": "#6c757d", "color": "#ffffff"}
                        )
                    ]
                ),
                # Label
                dbc.Row([
                    dbc.Col([
                        html.Small("Label", className="text-muted d-block fw-bold"),
                        html.Span(id = ids.NODE_INFO_LABEL, children="No node selected.", className="fs-6")
                    ], width=12),
                ], className="mb-3"),

                # ID & Samples
                dbc.Row([
                    dbc.Col([
                        html.Small("Node ID", className="text-muted d-block fw-bold"),
                        html.Span(id = ids.NODE_INFO_ID, children="—", className="fs-6")
                    ], width=2),
                    dbc.Col([
                        html.Small("Samples", className="text-muted d-block fw-bold"),
                        html.Span(id = ids.NODE_INFO_SAMPLES, children="—", className="fs-6")
                    ], width=2),
                    dbc.Col([
                        html.Small("Current RSS", className="text-muted d-block fw-bold"),
                        html.Span(id = ids.NODE_INFO_RSS, children="—", className="fs-6")
                    ], width=2),
                    dbc.Col([
                        html.Small("RSS root Reduction", className="text-muted d-block fw-bold"),
                        html.Span(id = ids.NODE_INFO_RSS_REDUCTION, children="—", className="fs-6")
                    ], width=3),
                ], className="mb-3")
                ,
                dbc.Switch(
                    id=ids.NODE_INFO_PLOT_SWITCH,
                    label="Show node plot",
                    value=False,
                ),
                dbc.Collapse(
                    id=ids.NODE_INFO_PLOT_COLLAPSE,
                    is_open=False,
                    children=[
                        dcc.Loading(
                            html.Div(
                                id=ids.NODE_INFO_PLOT_CONTAINER,
                                children="Plot will appear here, no node selected.",
                                style={"marginBottom": "8px"},
                            ),
                        ),
                        make_settings_card()
                    ]
                ),
            ]),
        ],
        class_name="mb-3",
    )
