"""
Consistent spinner treatment for slow actions. Fitting a new tree and
(later) computing SHAP-like explanations are the two genuinely slow
actions in this app -- most everything else (toggles, sliders, layout
tweaks) is fast and gets no spinner at all.
"""
import dash_bootstrap_components as dbc
from dash.development.base_component import Component

from config import SPINNER_COLOR


def with_spinner(component: Component, spinner_id: str) -> dbc.Spinner:
    """
    Wrap a component that a slow callback writes into. Target the *inner*
    component's id in your callback's Output.
    """
    return dbc.Spinner(component, id=spinner_id, color=SPINNER_COLOR, type="border")
