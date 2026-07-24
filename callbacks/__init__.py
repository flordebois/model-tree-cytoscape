"""
One place that wires up every callback module. app.py only calls
register_all_callbacks(app) once.

Pattern for a new page/feature: write register_callbacks(app) in a new
callbacks/xxx_callbacks.py, import it here, add one line below.
"""
from callbacks.tree_callbacks import register_callbacks as register_tree_callbacks
from callbacks.edit_tree_callbacks import register_callbacks as register_edit_tree_callbacks
from callbacks.node_callbacks import register_callbacks as register_node_callbacks
from callbacks.layout_callbacks import register_callbacks as register_layout_callbacks
from callbacks.highlight_callbacks import register_callbacks as register_highlight_callbacks
from callbacks.elements_callbacks import register_callbacks as register_elements_callbacks
from callbacks.explain_callbacks import register_callbacks as register_explain_callbacks


def register_all_callbacks(app):
    register_tree_callbacks(app)
    register_edit_tree_callbacks(app)
    register_node_callbacks(app)
    register_layout_callbacks(app)
    register_highlight_callbacks(app)
    #register_settings_callbacks(app)
    register_elements_callbacks(app)
    # register_explain_callbacks(app) #TODO: look at explain page
