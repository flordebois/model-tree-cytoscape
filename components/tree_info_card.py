"""
Tree Information card: pure display, no inputs. Shows which tree is
currently displayed (dataset, method, params). Split out of the old
"Tree" card so it can update independently and stay tiny/static.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
from config import get_initial_graph_info

import ids
from callbacks.tree_callbacks import format_tree_info

def make_tree_info_card() -> dbc.Card:
    _, tree_params = get_initial_graph_info()

    return dbc.Card(
        id=ids.TREE_INFO_TEXT,
        children=format_tree_info(**tree_params),
        class_name="mb-3"
    )
