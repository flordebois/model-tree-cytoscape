"""
Main workspace: Cytoscape graph + sidebar of six cards, in this order:
Tree Information, Node Information, New Tree, Layout, Highlight, Settings.

Registered as "/" (the home page). Pattern for adding a 7th card:
  1. components/x_card.py with make_x_card()
  2. an entry in ids.py for its controls
  3. callbacks/x_callbacks.py with register_callbacks(app)
  4. add make_x_card() to the sidebar list below + one line in callbacks/__init__.py
"""
import dash
import dash_bootstrap_components as dbc
from dash import html

import ids
from components.cytoscape_graph import make_cytoscape_graph
from components.tree_info_card import make_tree_info_card
from components.node_info_card import make_node_info_card
from components.new_tree_card import make_new_tree_card
from components.edit_tree_card import make_edit_tree_card
from components.layout_card import make_layout_card
from components.highlight_card import make_highlight_card
from components.node_info_settings_card import make_settings_card

dash.register_page(__name__, path="/", name="Tree View")

layout = dbc.Row(
    [
        dbc.Col(make_cytoscape_graph(), width=8, style={"height": "85vh"}),
        dbc.Col(
            [
                make_tree_info_card(),
                make_node_info_card(),
                make_new_tree_card(),
                make_edit_tree_card(),
                make_layout_card(),
                make_highlight_card(),
                # Low-key diagnostics; not a card, kept out of the way.
                html.Div(id=ids.DEBUG_INFO, style={"fontSize": "12px", "color": "#888"}), #TODO: Be consistent in where it is used
            ],
            width=4,
            style={"maxHeight": "85vh", "overflowY": "auto"},
        ),
    ]
)
