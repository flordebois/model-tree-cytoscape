from pathlib import Path

# --- Filesystem (used by app.py's image-serving Flask routes) ----------
DIR_BASE = Path(__file__).resolve().parent
DIR_LIVE_OUTPUT = DIR_BASE / "output" / "live"
DIR_SAVED_VIZ_TREES = DIR_BASE / "output" / "saved_viz_trees"
DIR_DATASETS = DIR_BASE / "datasets"

# --- New Tree card defaults / options -----------------------------------
# TODO: this list + PMLB_DATASETS_CAT_IDS lookup came from benchmark_info.py
# in the old single-file app -- bring that module back in once you've
# decided where dataset/category metadata should live.
DATASET_OPTIONS = [
    {"label": "547_no2", "value": "547_no2"},
    {"label": "294_satellite_image", "value": "294_satellite_image"},
    {"label": "1199_BNG_echoMonths", "value": "1199_BNG_echoMonths"},
    {"label": "658_fri_c3_250_25", "value": "658_fri_c3_250_25"},
    {"label": "505_tecator", "value": "505_tecator"},
    {"label": "560_bodyfat", "value": "560_bodyfat"},
    {"label": "485_analcatdata_vehicle", "value": "485_analcatdata_vehicle"},
    {"label": "210_cloud", "value": "210_cloud"},
    {"label": "1028_SWD", "value": "1028_SWD"},
    {"label": "simulated_linear", "value": "simulated_linear"},
    {"label": "california", "value": "california"},
    {"label": "197_cpu_act", "value": "197_cpu_act"},
]

NODE_COLORS = {
    "leafnode": "#2ca02c",
    "linearnode": "#9467bd",
    "blinnode": "#1f77b4",
    "pconnode": "#d62728",
    "plinnode": "#ff7f0e",
    "pconcnode": "#8c564b",
    "combinedlinnode": "#9467bd",
    "collapsednode": "#808080",
}

DEFAULT_DATASET_NAME = "547_no2"
DEFAULT_MAX_DEPTH = 12
DEFAULT_MAX_MODEL_DEPTH = 30
DEFAULT_MIN_SAMPLE_SPLIT = 10
DEFAULT_MIN_SAMPLE_LEAF = 5

# --- Layout card defaults ------------------------------------------------
DEFAULT_RANK_SEP = 30
DEFAULT_NODE_SEP = 20

# --- Settings card / plot defaults ---------------------------------------
DEFAULT_COLLAPSE_LEVEL = 5
DEFAULT_DISPLAY_TYPE = "histogram"
DEFAULT_NMAX = 5
DEFAULT_FIG_W = 7
DEFAULT_FIG_H = 3

SPINNER_COLOR = "primary"

# ── Stylesheet builder ─────────────────────────────────────────────────────────
font_family = "Arial, sans-serif"
font_size = 12
selected_color = "#000000"
CYTOSCAPE_STYLESHEET = [
    # ── Default node ──────────────────────────────────────────────────────
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "text-valign": "center",
            "text-halign": "center",
            "text-wrap": "wrap",
            "text-max-width": "140px",
            "font-family": font_family,
            "font-size": f"{font_size}px",
            "color": "#ffffff",
            "background-color": "#000000",  # default, normally not used
            "shape": "rectangle",  # default, normally not used
            "width": "100px",
            "height": "40px",
            "padding": "6px",
            "border-width": "1.5px",
            "border-color": "#000000",
        },
    },
    # ── Default edge ──────────────────────────────────────────────────────
    {
        "selector": "edge",
        "style": {
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "target-arrow-color": "#000000",
            "line-color": "#000000",
            "width": 1.5,
            "label": "data(label)",
            "font-family": font_family,
            "font-size": f"{font_size}px",
            "color": "#000000",
            "text-background-color": "#ffffff",
            "text-background-opacity": 0.7,
            "text-background-padding": "2px",
        },
    },
    # ── specific nodes ────────────────────────────────────────────────────
    {
        "selector": "node.LeafNode",
        "style": {
            "background-color": "#2ca02c",
            "shape": "ellipse",
            "width": "160px",
            "height": "60px",
        },
    },
    {
        "selector": "node.LinearNode",
        "style": {
            "background-color": "#9467bd",
            "shape": "ellipse",
        },
    },
    {
        "selector": "node.BlinNode",
        "style": {
            "background-color": "#1f77b4",
            "shape": "roundrectangle",
        },
    },
    {
        "selector": "node.PconNode",
        "style": {
            "background-color": "#d62728",
            "shape": "roundrectangle",
        },
    },
    {
        "selector": "node.PlinNode",
        "style": {
            "background-color": "#ff7f0e",
            "shape": "roundrectangle",
        },
    },
    {
        "selector": "node.PconcNode",
        "style": {
            "background-color": "#8c564b",
            "shape": "roundrectangle",
        },
    },
    {
        "selector": "node.CombinedLinNode",
        "style": {
            "background-color": "#9467bd",
            "shape": "ellipse",
        },
    },
    {
        "selector": "node.CollapsedNode",
        "style": {
            "background-color": "#808080",
        },
    },
    {
        "selector": "node.regplot",
        "style": {
            "label": "",
            "background-image": 'data(dir_regplot)',
            "shape": "rectangle",
            "width": "500px",
            "height": "300px",
        },
    },
    {
        "selector": "node.predsplot",
        "style": {
            "label": "",
            "background-image": 'data(dir_predsplot)',
            "shape": "rectangle",
            "width": "500px",
            "height": "300px",
        },
    },
    {
        "selector": "node.minimal",
        "style": {
            "label": "data(label_minimal)",
            "width": "30px",
            "height": "15px",
        },
    },
    # ── specific edges ────────────────────────────────────────────────────
    {
        "selector": "edge.combine_lin",
        "style": {
            "mid-target-arrow-shape": "circle",
            "mid-target-arrow-color": "#9467bd",
        }
    },

    # ── Selected highlight ────────────────────────────────────────────────
    {
        "selector": "node:selected",
        "style": {
            "border-width": "3px",
            "border-color": "#f0e442",
        },
    },
    # ── Highlighted path  ────────────────────────────────────────────────
    {
        "selector": "node.highlight",
        "style": {
            "border-width": "5px",
            "border-color": selected_color,
            "border-opacity": 1,
        },
    },
    {
        "selector": "edge.highlight",
        "style": {
            "line-color": selected_color,
            "target-arrow-color": selected_color,
            "width": 4,
        },
    },
]
