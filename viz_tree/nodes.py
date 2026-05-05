from abc import ABC, abstractmethod

import numpy as np
from typing import List, Tuple
import os
import matplotlib.pyplot as plt

from viz_tree.predsplot import predsplot
from viz_tree.dot_settings import DotSettings

NODE_SHAPES = {"leaf": "ellipse", "lin": "ellipse", "blin": "box", "pcon": "box", "plin": "box", "pconc": "box"}
NODE_LABEL = {"leaf": "Leaf", "lin": "LIN", "blin": "BLIN", "pcon": "PCON", "plin": "PLIN", "pconc": "PCONC"}
NODE_FILL_COLOR = {"leaf": "#2ca02c", "lin": "#9467bd", "blin": "#1f77b4", "pcon": "#d62728", "plin": "#ff7f0e", "pconc": "#8c564b"}
NODE_STYLE = {"leaf": "filled", "lin": "filled", "blin": "rounded,filled", "pcon": "rounded,filled", "plin": "filled", "pconc": "rounded,filled"}
NODE_FONT_NAME = "Arial"
NODE_FONT_COLOR = "white"

class BaseNode(ABC):
    id: int = None
    type: str = None
    X: np.ndarray = None
    y_res: np.ndarray = None
    
    def set_id(self, id: int):
        self.id = id

    @abstractmethod
    def get_children(self):
        pass

    @abstractmethod
    def get_label(self) -> str:
        return NODE_LABEL[self.type]




class LeafNode(BaseNode):
    def __init__(self, X, y_res, coefficients, intercept):
        self.type = "leaf"
        self.X = X
        self.y_res = y_res
        self.coefficients = coefficients
        self.intercept = intercept

    def get_children(self) -> List[BaseNode]:
        return []

    def get_label(self) -> str:
        parts = []
        for i, coef in enumerate(self.coefficients):
            if coef != 0:
                parts.append(f"{coef:.3g}·X{i}")
        parts.append(f"{self.intercept:.3g}")
        return "Leaf\n" + " + ".join(parts)

class InternalNode(BaseNode):
    type: str
    X: np.ndarray
    y_res: np.ndarray
    pivot_idx: int
    pivot_value: float # None for 'lin' type
    left_lin_model: Tuple[float, float]
    left_child_node: 'BaseNode'
    right_lin_model: Tuple[float, float] # None for 'lin' type
    right_child_node: 'BaseNode' # None for 'lin' type
    id: int = None

    def __init__(self, type, X, y_res, pivot_idx, pivot_value, left_lin_model, left_child_node,
                 right_lin_model, right_child_node):
        self.type = type
        self.X = X
        self.y_res = y_res
        self.pivot_idx = pivot_idx
        self.pivot_value = pivot_value
        self.left_child_node = left_child_node
        self.left_lin_model = left_lin_model
        self.right_child_node = right_child_node
        self.right_lin_model = right_lin_model

    def get_children(self) -> List[BaseNode]:
        children = [self.left_child_node]
        if self.right_child_node is not None:
            children.append(self.right_child_node)
        return children

    def get_label(self) -> str:
        ntype = self.type.upper()
        if self.type == "lin":
            return f"{ntype}\nX{self.pivot_idx}"
        return f"{ntype}\nX{self.pivot_idx} > {self.pivot_value:.3g}"

    def get_dot(self, id):
        self.id = id

        if self.type == "lin":
            return (
                f'node{self.id}[shape={NODE_SHAPES[self.type]}, label=<{NODE_LABEL[self.type]}<BR/>X<SUB><FONT POINT-SIZE="9">{self.pivot_idx}</FONT></SUB>>,'
                f'fontcolor={NODE_FONT_COLOR}, fontname="{NODE_FONT_NAME}", fillcolor="{NODE_FILL_COLOR[self.type]}", style="{NODE_STYLE[self.type]}", '
                f'margin=0.01, width=1.1]')
        else:
            return (
                f'node{self.id}[shape={NODE_SHAPES[self.type]}, label=<{NODE_LABEL[self.type]}<BR/>X<SUB><FONT POINT-SIZE="9">{self.pivot_idx} </FONT></SUB>&gt; {self.pivot_value:.3g}>,'
                f'fontcolor={NODE_FONT_COLOR}, fontname="{NODE_FONT_NAME}", fillcolor="{NODE_FILL_COLOR[self.type]}", style="{NODE_STYLE[self.type]}", '
                f'margin = 0.1]')

    def make_regression_plot(self, directory_regplot_map, fig_size, feature_colors, feature_names, highlight_x=None):
        os.makedirs(directory_regplot_map, exist_ok=True)
        directory_regplot_file = os.path.join(directory_regplot_map, f"regplot_node{self.id}.svg")

        fig = plt.figure(figsize=fig_size, layout="constrained")
        plt.gca().ticklabel_format(scilimits=[-3, 4])
        w = np.ones(len(self.y_res))
        feature_idx = self.pivot_idx
        fig.patch.set_linewidth(2)
        fig.patch.set_edgecolor(feature_colors[feature_idx])
        if feature_names is None:
            feature_label = "$X_{" + f"{feature_idx}" + "}$"
        else:
            feature_label = feature_names[feature_idx]
        min_x = min(self.X[:, feature_idx])
        max_x = max(self.X[:, feature_idx])
        scaled_weights = (w.flatten() - np.mean(w)) * 100 + 10
        plt.scatter(self.X[:, feature_idx], self.y_res, s=scaled_weights, color='slategrey')
        if highlight_x is not None:
            matches = np.all(self.X == highlight_x, axis=1)
            idx_point = np.argmax(matches) if np.any(matches) else None
            if idx_point is None:
                plt.axvline(x=highlight_x[feature_idx], linestyle='--', color='r')
            else:
                plt.scatter(self.X[idx_point, feature_idx], self.y_res[idx_point], s=60 + scaled_weights[idx_point],
                            facecolors='r', marker='*')

        if self.type == "lin":
            x = [min_x, max_x]
            y = [self.left_lin_model[1] + self.left_lin_model[0] * x for x in x]
            plt.plot(x, y, color=feature_colors[feature_idx], linewidth=3)  # color=NODE_FILL_COLOR[self.type]
            plt.title(f"LIN - Feature: {feature_label}")

        elif self.type == "pconc":
            pass

        else:  # self.type == "pcon", "plin", "blin"
            pivot = self.pivot_value
            x1 = [min_x, pivot]
            x2 = [pivot, max_x]
            y1 = [self.left_lin_model[1] + self.left_lin_model[0] * x for x in x1]
            y2 = [self.right_lin_model[1] + self.right_lin_model[0] * x for x in x2]
            plt.plot(x1, y1, x2, y2, color=feature_colors[feature_idx], linewidth=3)
            self_name = str(self.type).upper()
            plt.title(f"{self_name} - Feature: {feature_label} - Pivot: {pivot:.3g}")

        plt.xlabel(feature_label)
        y_label = "y" if self.id == 0 else "Residuals"
        plt.ylabel(y_label)
        plt.savefig(directory_regplot_file)
        plt.close()
        return directory_regplot_file

class CombinedLinNode(BaseNode):
    def __init__(self, nodes: List[InternalNode]):
        pivot_indices = []
        lin_coefficients = {}
        intercept = 0
        prev_node = None
        for node in nodes:
            # check that nodes are a chain of linear nodes
            if node.type != "lin":
                raise ValueError("One of the nodes is not linear")
            if prev_node is not None:
                if prev_node.left_child_node != node:
                    raise ValueError("Nodes are not a chain of linear nodes")
            prev_node = node

            pivot_indices.append(node.pivot_idx)
            lin_coefficients[node.pivot_idx] = node.left_lin_model[0]
            intercept += node.left_lin_model[1]

        self.child = nodes[-1].left_child_node
        self.pivot_indices = pivot_indices
        self.lin_coefficients = lin_coefficients
        self.intercept = intercept
        self.type = "combined_lin"
        self.X = nodes[-1].X
        self.y_res = nodes[-1].y_res

    def get_children(self):
        return [self.child]

    def get_label(self) -> str:
        idx_str = ", ".join(str(i) for i in self.pivot_indices)
        return f"LIN\nidx={idx_str}"