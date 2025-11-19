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
    node_id: int = None
    type: str = None

    @abstractmethod
    def get_dot(self, node_id) -> str:
        pass

    @abstractmethod
    def get_children(self):
        pass


class LeafNode(BaseNode):
    def __init__(self, X, y_res, coefficients, intercept):
        self.type = "leaf"
        self.X = X
        self.y_res = y_res
        self.coefficients = coefficients
        self.intercept = intercept

    def get_children(self) -> List[BaseNode]:
        return []

    def get_dot(self, node_id) -> str:
        self.node_id = node_id
        return (
            f'node{self.node_id}[shape={NODE_SHAPES[self.type]}, label=<{NODE_LABEL[self.type]}>, fontcolor={NODE_FONT_COLOR}, '
            f'fontname="{NODE_FONT_NAME}", fillcolor="{NODE_FILL_COLOR[self.type]}", style="{NODE_STYLE[self.type]}", '
            f'margin=0.01, width=0.9]')

    def get_dot_predsplot(self, node_id, directory_predsplot_map, dot_set: DotSettings, highlight) -> str:
        self.node_id = node_id

        os.makedirs(directory_predsplot_map, exist_ok=True)
        directory_predsplot_file = os.path.join(directory_predsplot_map, f"predsplot_node{self.node_id}.svg")

        y_hat = np.sum(self.coefficients * self.X, axis=1) + self.intercept
        highlight_x = dot_set.highlight_x if highlight else None
        intercept = self.intercept if dot_set.use_intercept else None
        predsplot(self.X, self.coefficients, y_hat, n_max=dot_set.n_max, intercept=intercept, fig_size=dot_set.fig_size,
                  feature_names = dot_set.feature_names, display_type=dot_set.display_type, truncate_total_pred=dot_set.truncate_total_pred, variable_tick_width=dot_set.variable_tick_width,
                  file_directory=directory_predsplot_file, highlight_x=highlight_x, staircase=dot_set.staircase)
        if highlight:
            return (f'node{self.node_id}[shape = box, width={dot_set.fig_size[0] + 0.2},'
                    f' height={dot_set.fig_size[1] + 0.2},'
                    f' label="", image="{directory_predsplot_file}", penwidth=3]')
        else:
            return (f'node{self.node_id}[shape = none, width={dot_set.fig_size[0] + 0.2},'
                    f' height={dot_set.fig_size[1] + 0.2},'
                    f' label="",image="{directory_predsplot_file}"]')


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
    node_id: int = None

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

    def get_dot(self, node_id):
        self.node_id = node_id

        if self.type == "lin":
            return (
                f'node{self.node_id}[shape={NODE_SHAPES[self.type]}, label=<{NODE_LABEL[self.type]}<BR/>X<SUB><FONT POINT-SIZE="9">{self.pivot_idx}</FONT></SUB>>,'
                f'fontcolor={NODE_FONT_COLOR}, fontname="{NODE_FONT_NAME}", fillcolor="{NODE_FILL_COLOR[self.type]}", style="{NODE_STYLE[self.type]}", '
                f'margin=0.01, width=1.1]')
        else:
            return (
                f'node{self.node_id}[shape={NODE_SHAPES[self.type]}, label=<{NODE_LABEL[self.type]}<BR/>X<SUB><FONT POINT-SIZE="9">{self.pivot_idx} </FONT></SUB>&gt; {self.pivot_value:.3g}>,'
                f'fontcolor={NODE_FONT_COLOR}, fontname="{NODE_FONT_NAME}", fillcolor="{NODE_FILL_COLOR[self.type]}", style="{NODE_STYLE[self.type]}", '
                f'margin = 0.1]')

    def get_dot_regplot(self, node_id, directory_regplot_map, dot_set: DotSettings, highlight):
        self.node_id = node_id

        os.makedirs(directory_regplot_map, exist_ok=True)
        directory_regplot_file = os.path.join(directory_regplot_map, f"regplot_node{self.node_id}.svg")

        plt.figure(figsize=dot_set.fig_size, layout="constrained")
        plt.gca().ticklabel_format(scilimits=[-3, 4])
        w = np.ones(len(self.y_res))
        feature_idx = self.pivot_idx
        if dot_set.feature_names is None:
            feature_label = "$X_{" + f"{feature_idx}" + "}$"
        else:
            feature_label = dot_set.feature_names[feature_idx]
        min_x = min(self.X[:, feature_idx])
        max_x = max(self.X[:, feature_idx])
        scaled_weights = (w.flatten() - np.mean(w)) * 100 + 10
        plt.scatter(self.X[:, feature_idx], self.y_res, s=scaled_weights, color='slategrey')
        if highlight:
            matches = np.all(self.X == dot_set.highlight_x, axis=1)
            idx_point = np.argmax(matches) if np.any(matches) else None
            if idx_point is None:
                plt.axvline(x=dot_set.highlight_x[feature_idx], linestyle='--', color='r')
            else:
                plt.scatter(self.X[idx_point, feature_idx], self.y_res[idx_point], s=60 + scaled_weights[idx_point],
                            facecolors='r', marker='*')

        if self.type == "lin":
            x = [min_x, max_x]
            y = [self.left_lin_model[1] + self.left_lin_model[0] * x for x in x]
            plt.plot(x, y, color=NODE_FILL_COLOR[self.type], linewidth=3)
            plt.title(f"Node: LIN - Feature: {feature_label}")

        elif self.type == "pconc":
            pass

        else:  # self.type == "pcon", "plin", "blin"
            pivot = self.pivot_value
            x1 = [min_x, pivot]
            x2 = [pivot, max_x]
            y1 = [self.left_lin_model[1] + self.left_lin_model[0] * x for x in x1]
            y2 = [self.right_lin_model[1] + self.right_lin_model[0] * x for x in x2]
            plt.plot(x1, y1, x2, y2, color=NODE_FILL_COLOR[self.type], linewidth=3)
            node_name = str(self.type).upper()
            plt.title(f"Node: {node_name} - Feature: {feature_label} - Pivot: {pivot:.3g}")

        plt.xlabel(feature_label)
        y_label = "y" if self.node_id == 0 else "Residuals"
        plt.ylabel(y_label)
        plt.savefig(directory_regplot_file)
        plt.close()

        if highlight:
            return (f'node{self.node_id}[shape = box, width={dot_set.fig_size[0] + 0.2},'
                    f' height={dot_set.fig_size[1]+ 0.2},'
                    f' label="", image="{directory_regplot_file}", penwidth=3]')
        else:
            return (f'node{self.node_id}[shape = none, width={dot_set.fig_size[0] + 0.2},'
                    f' height={dot_set.fig_size[1] + 0.2},'
                    f' label="",image="{directory_regplot_file}"]')

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


    def get_children(self):
        return [self.child]

    def get_dot(self, node_id):
        self.node_id = node_id
        indices_str = ','.join(map(str, self.pivot_indices))

        # Function to split string at comma boundaries
        def split_at_comma(s, max_len):
            if len(s) <= max_len:
                return s, ""

            last_comma = s[:max_len].rfind(',')
            if last_comma == -1:
                return split_at_comma(s, max_len + 1)

            return s[:last_comma + 1], s[last_comma + 1:]

        # Split the string into parts
        part1, rest = split_at_comma(indices_str, 12)
        parts = [part1]

        while len(rest) != 0:
            part, rest = split_at_comma(rest, 16)
            parts.append(part)

        # Build the HTML table
        table = '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="-1">\n'
        table += '    <TR>\n'
        table += '        <TD>LIN - X<SUB><FONT POINT-SIZE="9">idx</FONT></SUB></TD>\n'
        table += '    </TR>\n'
        for i, part in enumerate(parts):
            prefix = "idx=" if i == 0 else ""
            table += '    <TR>\n'
            table += f'        <TD>{prefix}{part}</TD>\n'
            table += '    </TR>\n'
        table += '</TABLE>'

        return (f'node{self.node_id}[shape={NODE_SHAPES["lin"]}, label=<{table}>,'
                f'fontcolor={NODE_FONT_COLOR}, fontname="{NODE_FONT_NAME}", '
                f'fillcolor="{NODE_FILL_COLOR["lin"]}", '
                f'style="{NODE_STYLE["lin"]}", margin=0, width=1.3]')