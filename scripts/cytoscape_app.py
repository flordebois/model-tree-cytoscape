import os
from pyexpat.errors import messages

from viz_tree import nodes

print("importing PILOT...")
from pilot.pilot import PILOT

print("import done")

import numpy as np
from pmlb import fetch_data
import time
import pickle

from dash import Dash, Input, Output, State, dcc, html, ctx
import dash_cytoscape as cyto
import dash_daq as daq
from flask import send_from_directory

cyto.load_extra_layouts()

from viz_tree.viz_tree_cytoscape import to_cytoscape_elements, build_cytoscape_stylesheet, highlight_path_for_x
from viz_tree.viz_tree import VizTree


def get_viz_tree(dataset_name, max_depth, max_model_depth, min_sample_split, min_sample_leaf, initial=False):
    if initial: #Initial bypass
        with open("/Users/flor/Pycharm/PILOT-VIS/scripts/output/saved_viz_trees/tree-547_no2-3-30-10-5.pkl", "rb") as f:
            dict_viz_tree = pickle.load(f)
            viz_tree = VizTree.from_dict(dict_viz_tree)
        return viz_tree
    print('Fitting model dataset:')
    print(dataset_name, max_depth, max_model_depth, min_sample_split, min_sample_leaf)
    data = fetch_data(dataset_name)
    X = np.array(data.iloc[:, :-1])
    y = np.array(data.iloc[:, -1])
    pilot_model = PILOT(max_depth=max_depth,
                        max_model_depth=max_depth,
                        min_sample_split=min_sample_split,
                        min_sample_leaf=min_sample_leaf)
    pilot_model.fit(X, y)
    pilot_tree = pilot_model.model_tree
    viz_tree = VizTree(pilot_tree, X, y, output_directory="output")
    return viz_tree

default_dataset_name = "547_no2"
default_max_depth = 3
default_max_model_depth = 30
default_min_sample_split = 10
default_min_sample_leaf = 5

initial_viz_tree      = get_viz_tree(default_dataset_name, default_max_depth, default_max_model_depth, default_min_sample_split, default_min_sample_leaf, initial=True)
initial_base_elements = to_cytoscape_elements(initial_viz_tree)
initial_stylesheet    = build_cytoscape_stylesheet()
print("Initial Finished")

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
                        "height": "1000px",
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
                            boxSelectionEnabled=True,
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
                                html.Div("PILOT", style={"fontWeight": "bold", "marginBottom": "10px"}),

                                html.Div("Shown tree information", style={"fontWeight": "bold", "marginBottom": "5px"}),
                                html.Div(
                                    id="tree-info",
                                    style={"marginBottom": "10px"},
                                    children= [(
                                        f"Dataset: {default_dataset_name}, "
                                        f"Max depth: {default_max_depth}, "
                                        f"Max model depth: {default_max_model_depth}, "
                                        f"Min sample split: {default_min_sample_split}, "
                                        f"Min sample leaf: {default_min_sample_leaf}")
                                    ]
                                ),

                                html.Div("New Pilot tree", style={"fontWeight": "bold", "marginBottom": "5px"}),
                                html.Div(
                                    style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "10px"},
                                    children=[
                                        html.Div([
                                            html.Div("Dataset", style={"marginBottom": "4px"}),
                                            dcc.Dropdown(
                                                id="input-dataset",
                                                options=[
                                                    {"label": "547_no2", "value": "547_no2"},
                                                    {"label": "294_satellite_image", "value": "294_satellite_image"},
                                                    {"label": "1199_BNG_echoMonths", "value": "1199_BNG_echoMonths"},
                                                    {"label": "658_fri_c3_250_25", "value": "658_fri_c3_250_25"},
                                                ],
                                                value="547_no2",
                                                clearable=False,
                                            )
                                        ], style={"marginBottom": "4px"}),
                                        html.Div([
                                            html.Div("Max depth", style={"marginBottom": "4px"}),
                                            dcc.Input(id="input-max-depth", type="number", value=12)
                                        ]),
                                        html.Div([
                                            html.Div("Max model depth", style={"marginBottom": "4px"}),
                                            dcc.Input(id="input-max-model-depth", type="number", value=30)
                                        ]),
                                        html.Div([
                                            html.Div("Min sample split", style={"marginBottom": "4px"}),
                                            dcc.Input(id="input-min-sample-split", type="number", value=10)
                                        ]),
                                        html.Div([
                                            html.Div("Min sample leaf", style={"marginBottom": "4px"}),
                                            dcc.Input(id="input-min-sample-leaf", type="number", value=5)
                                        ]),
                                    ],
                                ),
                                html.Div(
                                    style={"display": "flex", "gap": "8px", "marginBottom": "10px"},
                                    children=[
                                        html.Button(
                                            "Fit PILOT tree",
                                            id="btn-fit-pilot",
                                            style={"padding": "6px 12px", "borderRadius": "6px", "border": "1px solid #ccc",
                                                   "cursor": "pointer", "background": "#eee"},
                                        ),
                                        html.Button(
                                            "Save shown tree",
                                            id="btn-save-tree",
                                            style={"padding": "6px 12px", "borderRadius": "6px", "border": "1px solid #ccc",
                                                   "cursor": "pointer", "background": "#eee"},
                                        ),
                                        html.Button(
                                            "Download shown tree",
                                            id="btn-save-tree-svg",
                                            style={"padding": "6px 12px", "borderRadius": "6px", "border": "1px solid #ccc",
                                                   "cursor": "pointer", "background": "#eee"},
                                        )

                                    ],
                                ),

                                html.Div(
                                    style={"display": "flex", "gap": "8px"},
                                    children=[
                                        dcc.Input(
                                            id="input-load-tree",
                                            type="text",
                                            placeholder="e.g. /Users/flor/Pycharm/PILOT-VIS/scripts/output/saved_viz_trees/...",
                                            style={
                                                "flex": "1",
                                                "padding": "6px 8px",
                                                "borderRadius": "6px",
                                                "border": "1px solid #ccc",
                                            },
                                        ),
                                        html.Button(
                                            "Load tree",
                                            id="btn-load-tree",
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
                                    dcc.Slider(id="rank-sep", min=5, max=100, step=1, value=30, marks=None),
                                ]),
                                html.Div([
                                    "nodeSep (horizontal gap)",
                                    dcc.Slider(id="node-sep", min=5, max=100, step=1, value=20, marks=None),
                                ]),
                                html.Div(
                                    style={"display": "flex", "gap": "10px", "flexWrap": "wrap"},
                                    children=[
                                        daq.BooleanSwitch(id="switch-combine-lin", label="Combine linear nodes"),
                                        daq.BooleanSwitch(id="switch-reg-plots", label="Show regression plots"),
                                        daq.BooleanSwitch(id="switch-preds-plots", label="Show prediction plots"),
                                        daq.BooleanSwitch(id="switch-minimal", label="Minimal nodes"),
                                    ],
                                ),

                            ],
                        ),

                        # --- Plot card ---
                        html.Div(
                            style={
                                "background": "#f9f9f9",
                                "padding": "12px",
                                "borderRadius": "8px",
                                "border": "1px solid #ddd",
                                "marginBottom": "12px",
                            },
                            children=[

                                html.Div("Plot settings",
                                         style={"fontWeight": "bold", "marginBottom": "10px"}),

                                # n_max
                                html.Div(
                                    style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "10px"},
                                    children=[
                                       html.Div([
                                           html.Div("Display type", style={"marginBottom": "4px"}),
                                           dcc.Dropdown(
                                               id="input-display-type",
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
                                    ]
                                ),

                                # switches
                                html.Div(
                                    style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "10px"},
                                    children=[
                                        dcc.Checklist(
                                            id="predsplot-options",
                                            options=[
                                                {"label": "Use intercept", "value": "intercept"},
                                                {"label": "Truncate total pred", "value": "truncate"},
                                                {"label": "Staircase (highlight only)", "value": "staircase"},
                                            ],
                                            value=["truncate"],
                                            inline=True,
                                        )
                                    ],
                                ),

                                html.Button(
                                    "Refit & show plots",
                                    id="btn_refit-plots",
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
                                        dcc.Input(
                                            id="input-highlight",
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
                                            "Random point",
                                            id="btn-random-point",
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

                                dcc.Checklist(
                                    id="highlight-options",
                                    options=[
                                        {"label": "Only show highlighted path", "value": "only_show_highlight"},
                                    ],
                                    value=[],
                                    inline=True,
                                )
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
                                html.Div(id="node-info", style={"color": "#555", "fontSize": "13px"}, children="Click a node for details"),
                            ]
                        ),
                        html.Div(id="debug-info", style={"marginTop": "8px", "fontSize": "12px", "color": "#888"}),
                    ],
                ),
            ],
        ),

        dcc.Store(id="elements-trigger", data="initial"),
        dcc.Store(id="store-viz-tree", data=initial_viz_tree.to_dict()),
        dcc.Store(id="store-elements", data=initial_base_elements),
        dcc.Store(id="store-node-click", data={"last_click": 0, "last_id": None}),
        dcc.Store(id="store-highlight-x", data=None),
        dcc.Store(id="shown-tree-params", data={
                "dataset": "547_no2",
                "max_depth": 12,
                "max_model_depth": 30,
                "min_sample_split": 10,
                "min_sample_leaf": 5
            }),

    ],
)


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks
# ══════════════════════════════════════════════════════════════════════════════

# ── Node click ───────────────────────────────────────────────────
@app.callback(
    Output("node-info", "children"),
    Output("tree-graph", "elements", allow_duplicate=True),
    Output("store-elements", "data", allow_duplicate=True),
    Output("store-viz-tree", "data", allow_duplicate=True),
    Output("store-node-click", "data"),
    Input("tree-graph", "tapNodeData"),
    State("store-node-click", "data"),
    State("store-elements", "data"),
    State("store-viz-tree", "data"),
    prevent_initial_call=True,
)
def handle_click(data, store, elements, dict_viz_tree):
    now = time.time()
    last_time = store.get("last_click", 0)
    last_id = store.get("last_id")

    node_id = data["id"]

    # DOUBLE CLICK
    if last_id == node_id and now - last_time < 0.4:
        viz_tree = VizTree.from_dict(dict_viz_tree)

        # find node in viz_tree
        iter_nodes = iter(viz_tree.nodes)
        root = next(iter_nodes)
        while f"node{root.id}" != node_id:
            root = next(iter_nodes)

        viz_tree.root_node = root
        viz_tree.nodes = viz_tree.collect_nodes()
        viz_tree.edges = viz_tree.collect_edges()
        print([node.id for node in viz_tree.nodes])
        print([(par.id, child.id) for par, child in viz_tree.edges])

        # collect subtree
        subtree_ids = set()
        def collect(node):
            subtree_ids.add(f"node{node.id}")
            for c in node.get_children():
                collect(c)
        collect(root)

        new_elements = [
            el for el in elements
            if (
                ("id" in el.get("data", {}) and el["data"]["id"] in subtree_ids)
                or
                ("source" in el.get("data", {}) and
                 el["data"]["source"] in subtree_ids and
                 el["data"]["target"] in subtree_ids)
            )
        ]
        return f"Subtree from {node_id}", new_elements, new_elements, viz_tree.to_dict(), {"last_click": 0, "last_id": None}

    # SINGLE CLICK
    lines = [
        f"{k}: {v}"
        for k, v in data.items()
        if not k.startswith("_")
    ]
    return " | ".join(lines), elements, elements, dict_viz_tree, {"last_click": now, "last_id": node_id}

# ── Highlight ────────────────────────────────────────────────────────────
@app.callback(
    Output("input-highlight", "value"),
    Input("btn-random-point", "n_clicks"),
    State("store-viz-tree", "data"),
    prevent_initial_call=True,
)
def set_random_point(_, dict_viz_tree):
    viz_tree = VizTree.from_dict(dict_viz_tree)
    X_train = viz_tree.X_train
    string = ", ".join(map(str, X_train[np.random.randint(X_train.shape[0])]))
    return string

@app.callback(
    Output("store-highlight-x", "data", allow_duplicate=True),
    Output("debug-info", "children", allow_duplicate=True),
    Output("elements-trigger", "data", allow_duplicate=True),
    Input("btn-highlight", "n_clicks"),
    Input("btn-highlight-reset", "n_clicks"),
    State("input-highlight", "value"),
    State("store-viz-tree", "data"),
    prevent_initial_call=True,
)
def highlight_path(_, __, input_highlight_x, dict_viz_tree):
    if ctx.triggered_id == "btn-highlight-reset" or input_highlight_x is None:
        highlight_x = None
    else:
        viz_tree = VizTree.from_dict(dict_viz_tree)
        n_features = viz_tree.X_train.shape[1]
        try:
            highlight_x = np.array([float(char.strip()) for char in input_highlight_x.split(",")])
        except ValueError as e:
            raise ValueError("Cound't read input for highlighting") from e
        if len(highlight_x) != n_features:
            raise ValueError(f"Highlight input has length {highlight_x.shape[0]} != {n_features}")

    return highlight_x, f"Highlighting path.", time.time()

# ── Plot switches ──────────────────────────────────────────────────────
@app.callback(
    Output("switch-combine-lin", "on"),
    Output("switch-reg-plots", "on"),
    Output("switch-preds-plots", "on"),
    Output("switch-minimal", "on"),
    Output("elements-trigger", "data", allow_duplicate=True),
    Input("btn_refit-plots", "n_clicks"),
    Input("switch-combine-lin", "on"),
    Input("switch-reg-plots", "on"),
    Input("switch-preds-plots", "on"),
    Input("switch-minimal", "on"),
    prevent_initial_call=True,
)
def refit_plots(_, s1, s2, s3, s4):
    if ctx.triggered_id == "btn_refit-plots":
        if not (s2 or s3):
            return False, True, True, False, time.time()
        else:
            return False, s2, s3, False, time.time()
    elif ctx.triggered_id == "switch-combine-lin":
        return s1, False, s3, s4, time.time()
    elif ctx.triggered_id == "switch-reg-plots":
        return False, s2, s3, False, time.time()
    elif ctx.triggered_id == "switch-preds-plots":
        return s1, s2, s3, False, time.time()
    elif ctx.triggered_id == "switch-minimal":
        return s1, False, False, s4, time.time()
    return s1, s2, s3, s4, time.time()


# ── Elements update ──────────────────────────────────────────────────────
@app.callback(
    Output("tree-graph", "elements", allow_duplicate=True),
    Output("store-elements", "data", allow_duplicate=True),
    Output("debug-info", "children", allow_duplicate=True),
    Input("elements-trigger", "data"),
    State("switch-combine-lin", "on"),
    State("switch-reg-plots", "on"),
    State("switch-preds-plots", "on"),
    State("switch-minimal", "on"),
    State("store-viz-tree", "data"),
    State("input-display-type", "value"),
    State("input-nmax", "value"),
    State("input-fig-w", "value"),
    State("input-fig-h", "value"),
    State("predsplot-options", "value"),
    State("store-highlight-x", "data"),
    State("highlight-options", "value"),
    prevent_initial_call=True,
)
def update_elements(_, combine_lin, use_regplots, use_predsplots, use_minimal, dict_viz_tree,
                    display_type, nmax, figw, figh, predsplot_options, read_highlight_x, highlight_options):
    use_intercept = "intercept" in predsplot_options
    truncate_total_pred = "truncate" in predsplot_options
    staircase = "staircase" in predsplot_options
    viz_tree = VizTree.from_dict(dict_viz_tree)

    only_show_highlight = "only_show_highlight" in highlight_options
    highlight_x = np.array(read_highlight_x) if read_highlight_x is not None else None

    elements = to_cytoscape_elements(viz_tree,
                                     combine_lin=combine_lin,
                                     use_regplots=use_regplots,
                                     use_predsplots=use_predsplots,
                                     fig_size = (figw, figh),
                                     predsplot_n_max = nmax,
                                     predsplot_use_intercept = use_intercept,
                                     predsplot_display_type = display_type,
                                     predsplot_truncate_total_pred = truncate_total_pred,
                                     predsplot_staircase = staircase,
                                     highlight_x = highlight_x,
                                     only_show_highlight=only_show_highlight)
    if use_minimal:
        for el in elements:
            if "data" in el and "id" in el["data"]:  # node
                classes = el.get("classes", "").split()
                classes.append("minimal")
                el["classes"] = " ".join(classes)

    return elements, elements, "Elements updated."

# ── Layout update ──────────────────────────────────────────────────────
@app.callback(
    Output("tree-graph", "layout", allow_duplicate=True),
    Output("debug-info", "children", allow_duplicate=True),
    Input("btn-layout", "n_clicks"),
    Input("rank-dir", "value"),
    Input("rank-sep", "value"),
    Input("node-sep", "value"),
    prevent_initial_call=True
)
def update_layout(_, rank_dir, rank_sep, node_sep):
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
    Output("store-viz-tree", "data", allow_duplicate=True),
    Output("debug-info", "children", allow_duplicate=True),
    Output("shown-tree-params", "data", allow_duplicate=True),
    Output("tree-info", "children", allow_duplicate=True),
    Output("store-highlight-x", "data", allow_duplicate=True),
    Output("elements-trigger", "data", allow_duplicate=True),
    Input("btn-load-tree", "n_clicks"),
    Input("btn-fit-pilot", "n_clicks"),
    State("input-load-tree", "value"),
    State("input-dataset", "value"),
    State("input-max-depth", "value"),
    State("input-max-model-depth", "value"),
    State("input-min-sample-split", "value"),
    State("input-min-sample-leaf", "value"),
    prevent_initial_call=True
)
def update_viz_tree(_, __,dir_load_tree, dataset_name, max_depth, max_model_depth, min_sample_split, min_sample_leaf):
    if ctx.triggered_id == "btn-load-tree":
        dir_load_tree_copy = dir_load_tree
        if dir_load_tree_copy is None or not os.path.exists(dir_load_tree_copy):
            folder = "/Users/flor/Pycharm/PILOT-VIS/scripts/output/saved_viz_trees"
            dir_load_tree_copy = max(
                (os.path.join(folder, f) for f in os.listdir(folder)),
                key=os.path.getctime
            )
        with open(dir_load_tree_copy, "rb") as f:
            dict_viz_tree = pickle.load(f)
        message = f"Tree loaded from {dir_load_tree_copy}"
        file_name = os.path.basename(dir_load_tree_copy)
        string = file_name.removeprefix("tree-").removesuffix(".pkl")
        dataset_name, max_depth, max_model_depth, min_sample_split, min_sample_leaf = string.split("-")
    else: # ctx.triggered_id == "btn-fit-pilot":
        viz_tree = get_viz_tree(dataset_name, max_depth, max_model_depth, min_sample_split, min_sample_leaf)
        dict_viz_tree = viz_tree.to_dict()
        message = "New Pilot tree fitted on the dataset."

    tree_params = {
        "dataset": dataset_name,
        "max_depth": int(max_depth),
        "max_model_depth": int(max_model_depth),
        "min_sample_split": int(min_sample_split),
        "min_sample_leaf": int(min_sample_leaf)
    }
    tree_info = (
        f"Dataset: {dataset_name}, "
        f"Max depth: {max_depth}, "
        f"Max model depth: {max_model_depth}, "
        f"Min sample split: {min_sample_split}, "
        f"Min sample leaf: {min_sample_leaf}"
    )
    return dict_viz_tree, message, tree_params, tree_info, None, time.time()

# ── Save Viz tree ──────────────────────────────────────────────────────
@app.callback(
    Output("debug-info", "children", allow_duplicate=True),
    Input("btn-save-tree", "n_clicks"),
    State("store-viz-tree", "data"),
    State("shown-tree-params", "data"),
    prevent_initial_call=True
)
def update_viz_tree(_, dict_viz_tree, tree_params):
    file_name = (
        "tree_"
        f"{tree_params['dataset']}-"
        f"{tree_params['max_depth']}-"
        f"{tree_params['max_model_depth']}-"
        f"{tree_params['min_sample_split']}-"
        f"{tree_params['min_sample_leaf']}"
        ".pkl"
    )
    dir_save = os.path.join("/Users/flor/Pycharm/PILOT-VIS/scripts/output/saved_viz_trees", file_name)
    with open(dir_save, 'wb') as handle:
        pickle.dump(dict_viz_tree, handle)
    return f"Tree saved to {dir_save}"

# ── Download figure tree ──────────────────────────────────────────────────────
@app.callback(
    Output("tree-graph", "generateImage"),
    Input("btn-save-tree-svg", "n_clicks"),
    State("shown-tree-params", "data"),
    prevent_initial_call=True
)
def gen(_, tree_params):
    file_name = (
        "tree_"
        f"{tree_params['dataset']}-"
        f"{tree_params['max_depth']}-"
        f"{tree_params['max_model_depth']}-"
        f"{tree_params['min_sample_split']}-"
        f"{tree_params['min_sample_leaf']}"
    )
    return {
        "type": "svg",
        "action": "download",
        "filename": file_name
    }

# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(debug=True)
