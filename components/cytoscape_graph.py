"""
The central Cytoscape graph.

NOTE: in the old app, `elements` and `stylesheet` were populated at import
time by fitting an initial tree (get_viz_tree(...) -> to_cytoscape_elements(...)).
That logic isn't ported here -- this starts empty and expects a callback
(e.g. on page load, or after the first "Fit PILOT tree" click) to populate
tree-graph.elements. See callbacks/tree_callbacks.py.
"""

import dash_cytoscape as cyto
from dash import dcc
from viz_tree.viz_tree import VizTree

cyto.load_extra_layouts()

from viz_tree.viz_tree_cytoscape import viz_tree_to_cytoscape_elements
import ids
from config import *

(DIR_LIVE_OUTPUT / "regplots").mkdir(parents=True, exist_ok=True)
(DIR_LIVE_OUTPUT / "predsplots").mkdir(parents=True, exist_ok=True)
DIR_SAVED_VIZ_TREES.mkdir(parents=True, exist_ok=True)

viz_tree_dict, _ = get_initial_graph_info()
initial_viz_tree      = VizTree.from_dict(viz_tree_dict)
initial_base_elements = viz_tree_to_cytoscape_elements(initial_viz_tree, str(DIR_LIVE_OUTPUT))
initial_stylesheet    = CYTOSCAPE_STYLESHEET

def make_cytoscape_graph() -> cyto.Cytoscape:
    return cyto.Cytoscape(
        id=ids.CYTOSCAPE_GRAPH,
        elements=initial_base_elements,
        stylesheet=initial_stylesheet,
        layout={
            "name": "dagre",
            "rankDir": "TB",
            "rankSep": DEFAULT_RANK_SEP,
            "nodeSep": DEFAULT_NODE_SEP,
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
    )
