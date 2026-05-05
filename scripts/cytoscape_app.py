# print("importing PILOT...")
# from pilot.pilot import PILOT
#
# print("import done")

import numpy as np
from pmlb import fetch_data
import time
import pickle

from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
import dash_cytoscape as cyto
import dash_daq as daq
from flask import send_from_directory
cyto.load_extra_layouts()

from viz_tree.viz_tree_cytoscape import to_cytoscape_elements, build_cytoscape_stylesheet, highlight_path_for_x
from viz_tree.viz_tree import VizTree


def get_viz_tree(dataset_name, max_depth, max_model_depth, min_sample_split, min_sample_leaf):
    if True: #Temporary bypass
        print(dataset_name, max_depth, max_model_depth, min_sample_split, min_sample_leaf)
        with open("/Users/flor/Pycharm/PILOT-VIS/scripts/assets/my_viz_tree.pkl", "rb") as f:
            viz_tree = pickle.load(f)
        return viz_tree
    data = fetch_data(dataset_name)
    X = np.array(data.iloc[:, :-1])
    y = np.array(data.iloc[:, -1])
    #feature_names = data.iloc[:, :-1].columns.tolist() # TODO: add feature names???
    pilot_model = PILOT(max_depth=max_depth,
                        max_model_depth=max_depth,
                        min_sample_split=min_sample_split,
                        min_sample_leaf=min_sample_leaf)
    pilot_model.fit(X, y)
    pilot_tree = pilot_model.model_tree
    viz_tree = VizTree(pilot_tree, X, y, output_directory="output")
    return viz_tree

default_dataset_name = "547_no2"
default_max_depth = 12
default_max_model_depth = 30
default_min_sample_split = 10
default_min_sample_leaf = 5

initial_viz_tree      = get_viz_tree(default_dataset_name, default_max_depth, default_max_model_depth, default_min_sample_split, default_min_sample_leaf)
initial_base_elements = to_cytoscape_elements(initial_viz_tree)
initial_stylesheet    = build_cytoscape_stylesheet()


# ══════════════════════════════════════════════════════════════════════════════
# Layout
# ══════════════════════════════════════════════════════════════════════════════

app = Dash(__name__)

@app.server.route("/internal_regplots/<path:filename>")
def serve_images(filename):
    return send_from_directory("/Users/flor/Pycharm/PILOT-VIS/scripts/output/live/regplots", filename)

@app.server.route("/internal_predsplots/<path:filename>")
def serve_images2(filename):
    return send_from_directory("/Users/flor/Pycharm/PILOT-VIS/scripts/output/live/predsplots", filename)

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "padding": "16px"},
    children=[
        html.H2("Visualize Pilot trees"),

        html.Div(
            style={
                "display": "flex",
                "alignItems": "flex-start",
            },
            children=[

                # LEFT COLUMN (graph)
                html.Div(
                    style={
                        "flex": "2",
                        "height": "800px",
                        "overflow": "hidden",
                    },
                    children=[
                        cyto.Cytoscape(
                            id="tree-graph",
                            elements=initial_base_elements,
                            stylesheet=initial_stylesheet,
                            layout={
                                "name": "dagre",
                                "rankDir": "TB",
                                "rankSep": 30,
                                "nodeSep": 20,
                                "animate": True,
                                "fit": True,
                            },
                            style={
                                "width": "99%",
                                "height": "99%",
                                "border": "1px solid #ddd",
                                "borderRadius": "6px",
                            },
                            minZoom=0.1,
                            maxZoom=10,
                        ),
                    ],
                ),

                # RIGHT COLUMN (controls)
                html.Div(
                    style={"flex": "1"},
                    children=[

                        # --- Dataset card ---
                        html.Div(
                            style={
                                "background": "#f9f9f9",
                                "padding": "12px",
                                "borderRadius": "8px",
                                "border": "1px solid #ddd",
                                "marginBottom": "12px",
                            },
                            children=[
                                html.Div([
                                    html.Div([
                                        html.Div("Dataset", style={"marginBottom": "4px"}),
                                        dcc.Dropdown(
                                            id="input-dataset",
                                            options=[
                                                {"label": "547_no2", "value": "547_no2"},
                                                {"label": "294_satellite_image", "value": "294_satellite_image"},
                                            ],
                                            value="547_no2",
                                            clearable=False,
                                        )
                                    ], style={"marginBottom": "4px"}),
                                    html.Div([
                                        html.Div("Maximum depth", style={"marginBottom": "4px"}),
                                        dcc.Input(id="input-max-depth", type="number", value=12)
                                    ]),
                                    html.Div([
                                        html.Div("Maximum model depth", style={"marginBottom": "4px"}),
                                        dcc.Input(id="input-max-model-depth", type="number", value=30)
                                    ]),
                                    html.Div([
                                        html.Div("Minimum sample split", style={"marginBottom": "4px"}),
                                        dcc.Input(id="input-min-sample-split", type="number", value=10)
                                    ]),
                                    html.Div([
                                        html.Div("Minimum sample leaf", style={"marginBottom": "4px"}),
                                        dcc.Input(id="input-min-sample-leaf", type="number", value=5)
                                    ]),
                                    html.Button(
                                        "Fit PILOT tree",
                                        id="btn-fit-pilot",
                                        style={"padding": "6px 12px", "borderRadius": "6px", "border": "1px solid #ccc",
                                               "cursor": "pointer", "background": "#eee"},
                                    ),
                                ], style={"display": "flex", "gap": "4%"})
                            ],
                        ),

                        # --- Controls card ---
                        html.Div(
                            style={
                                "background": "#f9f9f9",
                                "padding": "12px",
                                "borderRadius": "8px",
                                "border": "1px solid #ddd",
                                "marginBottom": "12px",
                            },
                            children=[
                                html.Div(
                                    style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "10px"},
                                    children=[
                                        html.Button(
                                            "Refit layout",
                                            id="btn-layout",
                                            style={
                                                "padding": "6px 12px",
                                                "borderRadius": "6px",
                                                "border": "1px solid #ccc",
                                                "cursor": "pointer",
                                                "background": "#eee",
                                            },
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "10px"},
                                    children=[
                                        "Display direction:",
                                        dcc.RadioItems(
                                            id="rank-dir",
                                            options=["Top Bottom", "Left Right"],
                                            value="Top Bottom",
                                            inline=True,
                                        ),
                                    ],
                                ),
                                html.Div([
                                    "rankSep (vertical gap)",
                                    dcc.Slider(id="rank-sep", min=5, max=100, step=1, value=80, marks=None),
                                ]),
                                html.Div([
                                    "nodeSep (horizontal gap)",
                                    dcc.Slider(id="node-sep", min=5, max=100, step=1, value=40, marks=None),
                                ]),
                                html.Div(
                                    style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "10px"},
                                    children=[
                                        daq.BooleanSwitch(id="switch-combine-lin", label="Combine linear nodes"),
                                        daq.BooleanSwitch(id="switch-reg-plots", label="Show regression plots"),
                                        daq.BooleanSwitch(id="switch-preds-plots", label="Show prediction plots"),
                                    ],
                                ),

                            ],
                        ),

                        # --- Predsplot card ---
                        html.Div(
                            style={
                                "background": "#f9f9f9",
                                "padding": "12px",
                                "borderRadius": "8px",
                                "border": "1px solid #ddd",
                                "marginBottom": "12px",
                            },
                            children=[

                                html.Div("Prediction plot settings",
                                         style={"fontWeight": "bold", "marginBottom": "10px"}),

                                # n_max
                                html.Div([
                                    html.Div([
                                        html.Div("Display type", style={"marginBottom": "4px"}),
                                        dcc.Dropdown(
                                            id="dropdown-display",
                                            options=[
                                                {"label": "Histogram", "value": "histogram"},
                                                {"label": "Density", "value": "density"},
                                            ],
                                            value="histogram",
                                            clearable=False,
                                        )
                                    ]),
                                    html.Div([
                                        html.Div("Number of variables", style={"marginBottom": "4px"}),
                                        dcc.Input(id="input-nmax", type="number", value=5)
                                    ]),
                                    html.Div([
                                        html.Div("Figure size (inches)", style={"marginBottom": "4px"}),
                                        html.Div([
                                            dcc.Input(id="input-fig-w", type="number", value=5),
                                            dcc.Input(id="input-fig-h", type="number", value=3),
                                        ], style={"display": "flex", "gap": "4%"})
                                    ]),

                                ], style={"display": "flex", "gap": "4%", "marginBottom": "10px"}),

                                # switches
                                html.Div(
                                    style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "10px"},
                                    children=[
                                        daq.BooleanSwitch(id="switch-intercept", label="Use intercept", on=False),
                                        daq.BooleanSwitch(id="switch-truncate", label="Truncate total pred", on=True),
                                        daq.BooleanSwitch(id="switch-staircase", label="Staircase (highlight only)",on=True),
                                        html.Button(
                                            "Refit Prediction plots",
                                            id="btn_refit-preds-plots",
                                            style={
                                                "padding": "6px 12px",
                                                "borderRadius": "6px",
                                                "border": "1px solid #ccc",
                                                "cursor": "pointer",
                                                "background": "#eee",
                                            },
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        # --- Highlight card ---
                        html.Div(
                            style={
                                "background": "#f9f9f9",
                                "padding": "12px",
                                "borderRadius": "8px",
                                "border": "1px solid #ddd",
                                "marginBottom": "12px",
                            },
                            children=[
                                html.Div("Highlight prediction path",
                                         style={"fontWeight": "bold", "marginBottom": "8px"}),

                                html.Div(
                                    style={"display": "flex", "gap": "8px"},
                                    children=[
                                        dcc.Input(
                                            id="sample-input",
                                            type="text",
                                            placeholder="e.g. 1.2, 0.5, 3.1",
                                            style={
                                                "flex": "1",
                                                "padding": "6px 8px",
                                                "borderRadius": "6px",
                                                "border": "1px solid #ccc",
                                            },
                                        ),
                                        html.Button(
                                            "Highlight",
                                            id="btn-highlight",
                                            style={
                                                "padding": "6px 12px",
                                                "borderRadius": "6px",
                                                "border": "1px solid #ccc",
                                                "cursor": "pointer",
                                                "background": "#eee",
                                            },
                                        ),
                                        html.Button(
                                            "Clear highlight",
                                            id="btn-highlight-reset",
                                            style={
                                                "padding": "6px 12px",
                                                "borderRadius": "6px",
                                                "border": "1px solid #ccc",
                                                "cursor": "pointer",
                                                "background": "#eee",
                                            },
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        # --- Node info card ---
                        html.Div(
                            style={
                                "background": "#f9f9f9",
                                "padding": "12px",
                                "borderRadius": "8px",
                                "border": "1px solid #ddd",
                                "marginBottom": "12px",
                            },
                            children=[
                                html.Div("Node information", style={"fontWeight": "bold", "marginBottom": "8px"}),
                                html.Div(id="node-info", style={"color": "#555", "fontSize": "13px"}),
                            ]
                        ),
                        html.Div(id="debug-info", style={"marginTop": "8px", "fontSize": "12px", "color": "#888"}),
                    ],
                ),
            ],
        ),

        #dcc.Store(id="base-elements-store"),
        dcc.Store(id="store-viz-tree"),
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks
# ══════════════════════════════════════════════════════════════════════════════

# @app.callback(
#     Output("base-elements-store", "data"),
#     Input("tree-graph", "elements"),
#     State("base-elements-store", "data"),
#     prevent_initial_call=False,
# )
# def initialize_store(elements, stored):
#     if stored is None and elements:
#         return elements
#     return no_update

# ── Show node info on click ───────────────────────────────────────────────────
@app.callback(
    Output("node-info", "children"),
    Input("tree-graph", "tapNodeData"),
)
def display_node_info(data):
    if not data:
        return "Click a node for details"
    lines = []
    for key, val in data.items():
        if key.startswith("_"):
            continue
        lines.append(f"{key}: {val}")
    return " | ".join(lines)


# ── Path highlight ────────────────────────────────────────────────────────────
# @app.callback(
#     Output("tree-graph", "elements", allow_duplicate=True),
#     Output("debug-info", "children", allow_duplicate=True),
#     Input("btn-highlight", "n_clicks"),
#     Input("btn-highlight-reset", "n_clicks"),
#     State("sample-input", "value"),
#     State("base-elements-store", "data"),
#     State("store-viz-tree", "data"),
#     prevent_initial_call=True,
# )
# def update_highlight(n_hl, n_reset, sample_text, base_elements, viz_tree):
#     """
#     When the user clicks 'Highlight', parse their sample and walk the tree.
#     When they click 'Reset highlight', restore base elements.
#     """
#
#     triggered = callback_context.triggered_id
#     if not base_elements:
#         return [], "No tree loaded yet", time.time()
#
#     if triggered == "btn-highlight-reset" or not sample_text:
#         return base_elements, "Highlight cleared", time.time()
#
#     # Parse the sample
#     try:
#         x = np.array([float(v.strip()) for v in sample_text.split(",")])
#     except ValueError:
#         return base_elements, "⚠ Could not parse sample – use comma-separated numbers", time.time()
#
#     # Highlight path
#     highlighted = highlight_path_for_x(viz_tree, x, base_elements)
#     return highlighted, f"Path highlighted for sample with {len(x)} features", time.time()

# ── Elements update ──────────────────────────────────────────────────────
@app.callback(
    Output("tree-graph", "elements", allow_duplicate=True),
    Output("debug-info", "children", allow_duplicate=True),
    Input("switch-combine-lin", "on"),
    Input("switch-reg-plots", "on"),
    Input("switch-preds-plots", "on"),
    Input("store-viz-tree", "data"),
    prevent_initial_call=True,
)
def update_elements(combine_lin, use_regplots, use_predsplots, viz_tree):
    elements = to_cytoscape_elements(viz_tree,
                                     combine_lin=combine_lin,
                                     use_regplots=use_regplots,
                                     use_predsplots=use_predsplots
                                     )
    return elements, "Elements updated."

# ── Layout update ──────────────────────────────────────────────────────
@app.callback(
    Output("tree-graph", "layout"),
    Output("debug-info", "children", allow_duplicate=True),
    Input("btn-layout", "n_clicks"),
    Input("rank-dir", "value"),
    Input("rank-sep", "value"),
    Input("node-sep", "value"),
    prevent_initial_call=True
)
def update_layout(data, rank_dir, rank_sep, node_sep):
    if data is None: data = 0
    return {
        "name": "dagre",
        "rankDir": rank_dir.split(" ")[0][0] + rank_dir.split(" ")[1][0],
        "rankSep": rank_sep,
        "nodeSep": node_sep,
        "animate": True,
        "fit": True,
        "_time": time.time()  # dummy changing field
    }, "Layout updated."

# ── Viz tree update ──────────────────────────────────────────────────────
@app.callback(
    Output("store-viz-tree", "data"),
    #Output("debug-info", "children", allow_duplicate=True),
    Input("input-dataset", "value"),
    Input("input-max-depth", "value"),
    Input("input-max-model-depth", "value"),
    Input("input-min-sample-split", "value"),
    Input("input-min-sample-leaf", "value"),
)
def update_viz_tree(dataset_name, max_depth, max_model_depth, min_sample_split, min_sample_leaf):
    new_viz_tree = get_viz_tree(dataset_name, max_depth, max_model_depth, min_sample_split, min_sample_leaf)
    return new_viz_tree#, "New Pilot tree fitted on the dataset."

# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(debug=True)
