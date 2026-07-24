"""
dcc.Store components carried over verbatim from the old app.layout.
Kept outside dash.page_container (see app.py) so Explain can still read
store-viz-tree without you having to re-plumb anything.
"""
from dash import dcc, html

import ids


def make_global_stores() -> html.Div:
    return html.Div(
        [
            dcc.Store(id=ids.ELEMENTS_TRIGGER, data=0),
            dcc.Store(id=ids.STORE_VIZ_TREE),
            dcc.Store(id=ids.STORE_VIZ_TREE_BASE),
            dcc.Store(id=ids.STORE_NODE_CLICK, data={"last_click": 0, "last_id": None}),
            dcc.Store(id=ids.STORE_HIGHLIGHT_X, data=None),
            dcc.Store(id=ids.STORE_TREE_PARAMS,data=None),
            dcc.Store(id=ids.STORE_TREE_PARAMS_BASE, data=None),
        ]
    )
