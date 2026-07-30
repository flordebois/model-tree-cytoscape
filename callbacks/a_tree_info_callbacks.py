import dash_bootstrap_components as dbc
from dash import html, Output, Input
import ids

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

def register_callbacks(app):
    # --- UPDATE TREE PARAMS ---
    @app.callback(
        Output(ids.TREE_INFO_TEXT, "children"),

        Input(ids.STORE_TREE_PARAMS, "data"),
        prevent_initial_call=True
    )
    def update_tree_info_card(tree_params):
        return format_tree_info(**tree_params)
