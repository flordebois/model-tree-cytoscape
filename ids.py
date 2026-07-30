# --- Cytoscape graph ----------------------------------------------------
CYTOSCAPE_GRAPH = "tree-graph"

# --- Tree Information card (new: split out of the old "Tree" card) ------
TREE_INFO_TEXT = "tree-info"

# --- Node Information card (new: split out; node-plot switch is NEW) ----
NODE_INFO_TYPE = "node-info-type"
NODE_INFO_LABEL = "node-info-label"
NODE_INFO_ID = "node-info-id"
NODE_INFO_SAMPLES = "node-info-samples"
NODE_INFO_RSS = "node-info-rss"
NODE_INFO_RSS_REDUCTION = "node-info-rss-reduction"
NODE_INFO_PLOT_SWITCH = "node-info-plot-switch"          # NEW
NODE_INFO_PLOT_COLLAPSE = "node-info-plot-collapse"      # NEW
NODE_INFO_PLOT_CONTAINER = "node-info-plot-container"    # NEW

# --- New Tree card (fit / load / save / reload) --------------------------
INPUT_DATASET = "input-dataset"
INPUT_METHOD = "input-method"
INPUT_MAX_DEPTH = "input-max-depth"
INPUT_MAX_MODEL_DEPTH = "input-max-model-depth"
INPUT_MIN_SAMPLE_SPLIT = "input-min-sample-split"
INPUT_MIN_SAMPLE_LEAF = "input-min-sample-leaf"
BTN_FIT_NEW_TREE = "btn-fit-new-tree"
BTN_SAVE_TREE = "btn-save-tree"
BTN_SAVE_TREE_SVG = "btn-save-tree-svg"
BTN_RELOAD_TREE = "btn-reload-tree"
INPUT_LOAD_TREE = "input-load-tree"
BTN_LOAD_TREE = "btn-load-tree"
NEW_TREE_FIT_OUTPUT = "new-tree-fit-output"              # NEW (feedback area)
NEW_TREE_FIT_SPINNER = "new-tree-fit-spinner"            # NEW

# --- Layout card ----------------------------------------------------------
BTN_LAYOUT = "btn-layout"
RANK_DIR = "rank-dir"
RANK_SEP = "rank-sep"
NODE_SEP = "node-sep"

# --- Highlight card ---------------------------------------------------------
BTN_HIGHLIGHT = "btn-highlight"
BTN_CLEAR_HIGHLIGHT = "btn-clear-highlight"
INPUT_HIGHLIGHT = "input-highlight"
BTN_RANDOM_POINT = "btn-random-point"
HIGHLIGHT_OPTIONS = "highlight-options"

# Tree structure section
BTN_COLLAPSE_EXPAND = "btn-collapse-expand"
BTN_SUBTREE = "btn-subtree"
BTN_EXPAND_ALL = "btn-expand-all"
BTN_COLLAPSE_LEVEL = "btn-collapse-level"
INPUT_COLLAPSE_LEVEL = "input-collapse-level"
COLLAPSE_LEVEL_OPTIONS = "collapse-level-options"

# Display-switches section
SWITCH_COMBINE_LIN = "switch-combine-lin"
SWITCH_NODE_PLOTS = "switch-node-plots"
SWITCH_MINIMAL = "switch-minimal"
SWITCH_COLOR_FEATURES = "switch-color-features"
SWITCH_RSS = "switch-rss"
MODAL_NODE_PLOTS = "modal-node-plots"
MODAL_BTN_CONFIRM = "modal-btn-confirm"
MODAL_BTN_CANCEL = "modal-btn-cancel"
MODAL_CHECK_DONT_ASK = "modal-dont-ask"

# Plot-settings section
INPUT_DISPLAY_TYPE = "input-display-type"
INPUT_NMAX = "input-nmax"
INPUT_FIG_W = "input-fig-w"
INPUT_FIG_H = "input-fig-h"
PREDSPLOT_OPTIONS = "predsplot-options"
PREDSPLOT_TYPE = "predsplot-type"
BTN_REFIT_PLOTS = "btn_refit-plots"

# --- Shared stores (defined in components/stores.py, live outside
#     dash.page_container so they survive navigation to /explain) ---------
ELEMENTS_TRIGGER = "elements-trigger"
STORE_VIZ_TREE = "store-viz-tree"
STORE_VIZ_TREE_BASE = "store-viz-tree-base"
STORE_NODE_CLICK = "store-node-click"
STORE_HIGHLIGHT_X = "store-highlight-x"
STORE_TREE_PARAMS = "store-tree-params"
STORE_TREE_PARAMS_BASE = "store-tree-params-base"
STORE_MODAL_DONT_ASK = "store-modal-dont-ask"

# --- Explain page ------------------------------------------------------------
EXPLAIN_COMPUTE_BUTTON = "explain-compute-button"
EXPLAIN_PLOT_TYPE_DROPDOWN = "explain-plot-type-dropdown"
EXPLAIN_OUTPUT_CONTAINER = "explain-output-container"

# --- Misc -------------------------------------------------------------------
DEBUG_INFO = "debug-info"
