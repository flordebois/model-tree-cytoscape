"""
Tree Information card: pure display, no inputs. Shows which tree is
currently displayed (dataset, method, params). Split out of the old
"Tree" card so it can update independently and stay tiny/static.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
from config import (DEFAULT_DATASET_NAME, DEFAULT_MAX_DEPTH, DEFAULT_MAX_MODEL_DEPTH,
                    DEFAULT_MIN_SAMPLE_SPLIT, DEFAULT_MIN_SAMPLE_LEAF)

import ids
from callbacks.tree_callbacks import format_tree_info

def make_tree_info_card() -> dbc.Card:
    return dbc.Card(
        id=ids.TREE_INFO_TEXT,
        children=dcc.Loading(format_tree_info(DEFAULT_DATASET_NAME, DEFAULT_MAX_DEPTH, DEFAULT_MAX_MODEL_DEPTH,
                                  DEFAULT_MIN_SAMPLE_SPLIT, DEFAULT_MIN_SAMPLE_LEAF, -1.0),
                             delay_hide = 1000
                             ),
        class_name="mb-3"
    )
