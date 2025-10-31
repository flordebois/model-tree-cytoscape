import numpy as np
import pandas as pd
from typing import List, Tuple
import hashlib
import json
import os
from datetime import datetime

from pilot.tree import tree
from pilot.predsplot import predsplot

NODE_SHAPES = {"leaf": "ellipse", "lin": "ellipse", "blin": "box", "pcon": "box", "plin": "box", "pconc": "box"}
NODE_LABEL = {"leaf": "Leaf", "lin": "LIN", "blin": "BLIN", "pcon": "PCON", "plin": "PLIN", "pconc": "PCONC"}
NODE_FILL_COLOR = {"leaf": "#2ca02c", "lin": "#9467bd", "blin": "#1f77b4", "pcon": "#d62728", "plin": "#ff7f0e", "pconc": "#8c564b"}
NODE_STYLE = {"leaf": "filled", "lin": "filled", "blin": "rounded,filled", "pcon": "rounded,filled", "plin": "filled", "pconc": "rounded,filled"}
NODE_FONT_NAME = "Arial"
NODE_FONT_COLOR = "white"

class Node:
    type: str
    pivot_idx: int
    pivot_value: float
    categorical: bool
    left_lin_model: Tuple[float, float]
    left_child_node: 'Node'
    right_lin_model: Tuple[float, float]
    right_child_node: 'Node'
    dot_id: int = None

    def __init__(self, type, pivot_idx, pivot_value, categorical, left_lin_model, left_child_node, right_lin_model, right_child_node):
        self.type = type
        self.pivot_idx = pivot_idx
        self.pivot_value = pivot_value
        self.categorical = categorical
        self.left_child_node = left_child_node
        self.left_lin_model = left_lin_model
        self.right_child_node = right_child_node
        self.right_lin_model = right_lin_model

    def get_dot(self, id):
        self.dot_id = id

        if self.type == "leaf":
            return (
                f'node{id}[shape={NODE_SHAPES[self.type]}, label=<{NODE_LABEL[self.type]}>, fontcolor={NODE_FONT_COLOR}, '
                f'fontname="{NODE_FONT_NAME}", fillcolor="{NODE_FILL_COLOR[self.type]}", style="{NODE_STYLE[self.type]}", '
                f'margin=0.01, width=0.9]')
        elif self.type == "lin":
            return (
                f'node{id}[shape={NODE_SHAPES[self.type]}, label=<{NODE_LABEL[self.type]}<BR/>X<SUB><FONT POINT-SIZE="9">{self.pivot_idx}</FONT></SUB>>,'
                f'fontcolor={NODE_FONT_COLOR}, fontname="{NODE_FONT_NAME}", fillcolor="{NODE_FILL_COLOR[self.type]}", style="{NODE_STYLE[self.type]}", '
                f'margin=0.01, width=1.1]')
        else:
            return (
                f'node{id}[shape={NODE_SHAPES[self.type]}, label=<{NODE_LABEL[self.type]}<BR/>X<SUB><FONT POINT-SIZE="9">{self.pivot_idx} </FONT></SUB>&gt; {np.round(self.pivot_value,2)}>,'
                f'fontcolor={NODE_FONT_COLOR}, fontname="{NODE_FONT_NAME}", fillcolor="{NODE_FILL_COLOR[self.type]}", style="{NODE_STYLE[self.type]}", '
                f'margin = 0.1]')


class CombinedLinNode(Node):
    def __init__(self, nodes: List[Node]):
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


        super().__init__(
            type="lin",
            pivot_idx=None,  # Not used for combined nodes
            pivot_value=None,  # Not used for combined nodes
            categorical=False,  # Linear nodes aren't categorical
            left_lin_model=(0,0),
            left_child_node=nodes[-1].left_child_node,
            right_lin_model=(0,0),
            right_child_node=None
        )
        self.pivot_indices = pivot_indices
        self.lin_coefficients = lin_coefficients
        self.intercept = intercept

    def get_dot(self, id):
        self.dot_id = id
        indices_str = ','.join(map(str, self.pivot_indices))

        # Function to split string at comma boundaries
        def split_at_comma(s, max_len):
            if len(s) <= max_len:
                return s, ""

            last_comma = s[:max_len].rfind(',')
            if last_comma == -1:
                return split_at_comma(s, max_len+1)

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

        return (
            f'node{id}[shape={NODE_SHAPES[self.type]}, label=<{table}>,'
            f'fontcolor={NODE_FONT_COLOR}, fontname="{NODE_FONT_NAME}", '
            f'fillcolor="{NODE_FILL_COLOR[self.type]}", '
            f'style="{NODE_STYLE[self.type]}", margin=0, width=1.3]'
        )

class LeafNodePredsplot(Node):
    X: np.ndarray
    coefficients: np.ndarray
    y_hat: np.ndarray
    intercept: np.float64
    directory_predsplot_map: str

    def __init__(self, X, coefficients, intercept, directory):

        if directory is None:
            raise Exception(f"No directory specified for predsplots")
        self.directory_predsplot_map = directory

        self.X = X
        self.coefficients = coefficients
        self.intercept = intercept
        self.y_hat = np.sum(coefficients*X, axis=1) + intercept

        super().__init__(
            type="leaf",
            pivot_idx=None,  # Not used for leaf nodes
            pivot_value=None,  # Not used for leaf nodes
            categorical=False,  # leaf nodes aren't categorical
            left_lin_model=(0,0),
            left_child_node=None,
            right_lin_model=(0,0),
            right_child_node=None
        )

    def get_dot(self, id):
        self.dot_id = id
        os.makedirs(self.directory_predsplot_map, exist_ok=True)
        directory_predsplot_file = os.path.join(self.directory_predsplot_map, f"predsplot_node{id}.svg")

        predsplot(self.X, self.coefficients, self.y_hat, n_max=5, fig_size=(5, 3), truncate_total_pred=True,
                  variable_tick_width=True, file_directory=directory_predsplot_file)
        # return (
        #     f'node{id}[shape={NODE_SHAPES[self.type]}, label=<LeafNodePredsplot>, fontcolor={NODE_FONT_COLOR}, '
        #     f'fontname="{NODE_FONT_NAME}", fillcolor="{NODE_FILL_COLOR[self.type]}", style="{NODE_STYLE[self.type]}", '
        #     f'margin=0.01, width=0.9]')

        return (
            f'node{id}[margin="0" shape=none label='
            f'<<table border="0"><tr><td><img src="{directory_predsplot_file}"/></td></tr></table>>]')

class VizTree:
    root_node: Node
    nodes: List[Node]
    edges: List[Tuple[Node, Node]]
    feature_names: List[str]
    target_name: str
    X_train: np.ndarray
    y_train: np.ndarray
    directory: str
    predsplot_leafs: bool

    def __init__(self,
                 tree_model: tree,
                 X_train: (pd.DataFrame, np.ndarray),
                 y_train: (pd.Series, np.ndarray),
                 feature_names: List[str] = None,
                 target_name: str = None,
                 predsplot_leafs: bool = False,
                 output_directory: str = None,
                 ):
        """
        Parameters
        ----------
        :param tree_model: Pilot model tree
            The model tree to be interpreted
        :param X_train: pd.DataFrame, np.ndarray
            Features values on which the tree is build.
        :param y_train: pd.Series, np.ndarray
            Target values on which the shadow tree is build.
        :param feature_names: List[str]
            Features' names
        :param target_name: str
            Target's name
        ...
        """

        self.feature_names = feature_names
        self.target_name = target_name
        if isinstance(X_train, pd.core.frame.DataFrame):
            self.X_train = np.array(X_train)
        else:
            self.X_train = X_train
        if isinstance(y_train, pd.core.frame.DataFrame):
            self.y_train = np.array(y_train)
        else:
            self.y_train = y_train

        self.root_node = get_root_node(tree_model)
        self.predsplot_leafs = predsplot_leafs

        if output_directory is not None:
            directory_map = os.path.join(output_directory, "predsplots")
            os.makedirs(directory_map, exist_ok=True)
            tree_id = datetime.now().strftime('%d-%m-%y_%H-%M-%S')
            self.directory_predsplot_map = os.path.join(directory_map, f"tree_id_{tree_id}")

        if self.predsplot_leafs:
            self.root_node = self._make_leaf_nodes_predsplot(self.root_node, np.ones(self.X_train.shape[0], dtype=bool),
                                                             np.zeros(self.X_train.shape[1]), 0)

        self.nodes = get_nodes_list(self.root_node)

        edges = []
        for node in self.nodes:
            if node.left_child_node is not None:
                edges.append((node, node.left_child_node))
            if node.right_child_node is not None:
                edges.append((node, node.right_child_node))
        self.edges = edges


    def _make_leaf_nodes_predsplot(self, node: Node, indices: np.ndarray, coefficients: np.ndarray, intercept: float) -> Node|None:
        """Recursively traverse and replace leaf nodes with LeafNodePredsplot"""
        if node is None:
            return None

        if node.type == 'leaf':
            new_leaf = LeafNodePredsplot(self.X_train[indices,:], coefficients, intercept, self.directory_predsplot_map)
            return new_leaf
        else:
            if node.type == 'lin':
                left_indices = indices
                right_indices = None
            else:
                left_of_pivot = self.X_train[:,node.pivot_idx] <= node.pivot_value
                left_indices = indices & left_of_pivot
                right_indices = indices & (~left_of_pivot)

            left_coefficients = coefficients.copy()
            right_coefficients = coefficients.copy()
            left_coefficients[node.pivot_idx] += node.left_lin_model[0]
            right_coefficients[node.pivot_idx] += node.right_lin_model[0]

            left_intercept = intercept + node.left_lin_model[1]
            right_intercept = intercept + node.right_lin_model[1]

            node.left_child_node = self._make_leaf_nodes_predsplot(node.left_child_node, left_indices, left_coefficients, left_intercept)
            node.right_child_node = self._make_leaf_nodes_predsplot(node.right_child_node, right_indices, right_coefficients, right_intercept)
            return node

    def get_dot(self, combine_lin=False, x=None):
        # Find nodes and edges to highlight along path for x
        highlight_nodes = []
        highlight_edges = []
        if x is not None:
            node_in_path = self.root_node
            highlight_nodes.append(node_in_path)
            while node_in_path.type != "leaf":
                parent_node_in_path = node_in_path
                if node_in_path.type == "lin":
                    node_in_path = node_in_path.left_child_node
                else:
                    if x[node_in_path.pivot_idx] > node_in_path.pivot_value:
                        node_in_path = node_in_path.right_child_node
                    else:
                        node_in_path = node_in_path.left_child_node
                highlight_nodes.append(node_in_path)
                highlight_edges.append((parent_node_in_path, node_in_path))
            node_in_path.get_dot = lambda self: "OVERRIDDEN OUTPUT" #Does not work?

        # Combine multiple linear into one
        if combine_lin:
            nodes = []
            edges = self.edges.copy()
            new_node_mapping = {}
            for node in self.nodes:
                new_node_mapping[node] = node
            for node in self.nodes:
                if node.type == "lin" and node.left_child_node.type == "lin" and new_node_mapping[node] == node:
                    combine_nodes = [node]
                    next_node = node.left_child_node
                    while next_node.type == "lin":
                        combine_nodes.append(next_node)
                        next_node = next_node.left_child_node

                    combined_lin_node = CombinedLinNode(combine_nodes)
                    nodes.append(combined_lin_node)
                    for node in combine_nodes:
                        new_node_mapping[node] = combined_lin_node
                    if node in highlight_nodes:
                        highlight_nodes.append(combined_lin_node)
                elif node.type == "lin" and new_node_mapping[node] != node:
                    continue
                else:
                    nodes.append(node)

            edges = []
            for edge in self.edges:
                new_parent_node = new_node_mapping[edge[0]]
                new_child_node = new_node_mapping[edge[1]]
                if new_parent_node != new_child_node:
                    edges.append((new_parent_node, new_child_node))
                    if edge in highlight_edges:
                        highlight_edges.append((new_parent_node, new_child_node))

        else:
            # No combining of linear nodes
            nodes = self.nodes
            edges = self.edges

        # Get .dot for each node
        nodes_dot = []
        for i, node in enumerate(nodes):
            if node in highlight_nodes:
                dot = node.get_dot(i)[:-1] + ", penwidth=3]"
                nodes_dot.append(dot)
            else:
                nodes_dot.append(node.get_dot(i))

        # Get .dot for each edge
        edges_dot = []
        for edge in edges:
            parent_node, child_node = edge
            if edge in highlight_edges:
                if isinstance(parent_node, CombinedLinNode):
                    linear_label = ""
                    for pivot_idx in parent_node.lin_coefficients.keys():
                        linear_label += f'{np.round(parent_node.lin_coefficients[pivot_idx],2)}X<SUB><FONT POINT-SIZE="9">{pivot_idx}</FONT></SUB> + '
                    label = f'{linear_label}{np.round(parent_node.intercept,2)}'
                else:
                    if parent_node.left_child_node == child_node:
                        linear_model = parent_node.left_lin_model
                    else:
                        linear_model = parent_node.right_lin_model
                    if parent_node.type == "pcon" or parent_node.type == "pconc":
                        label = f'{np.round(linear_model[1], 2)}'
                    else:
                        label = f'{np.round(linear_model[0], 2)}X<SUB><FONT POINT-SIZE="9">{parent_node.pivot_idx}</FONT></SUB> + {np.round(linear_model[1], 2)}'
                edges_dot.append(f'node{parent_node.dot_id} -> node{child_node.dot_id}'
                                 f'[penwidth=3 fontname={NODE_FONT_NAME}, label=<{label}>]')
            else:
                edges_dot.append(f"node{parent_node.dot_id} -> node{child_node.dot_id}")

        #Combine .dot of nodes and edges
        newline = "\n\t"
        dot = \
            f"""
                digraph G {{
                    splines=line;
                    rankdir=TD;

                    {newline.join(nodes_dot)}
                    {newline.join(edges_dot)}
                }}
            """

        return dot



def get_tree_nodes(tree_model: tree) -> Tuple[Node, List[Node]]:
    root_node = get_root_node(tree_model)

    return root_node, get_nodes_list(root_node)

def get_nodes_list(node:Node) -> List[Node]:
    if node is None:
        return []
    else:
        return ([node] +
                get_nodes_list(node.left_child_node) +
                get_nodes_list(node.right_child_node))

def get_root_node(tree_model: tree) -> Node:
    if tree_model is None:
        raise ValueError("Tree model is None")
    if tree_model.node == 'con' or tree_model.node == 'END':
        return Node('leaf',0, 0, False, (0,0), None, (0,0), None)
    elif tree_model.node == 'lin':
        pivot_idx, _ = tree_model.pivot
        child_node = get_root_node(tree_model.left)
        return Node('lin',pivot_idx, 0, False, tree_model.lm_l, child_node, (0,0), None)
    elif tree_model.node == 'pconc':
        pivot_idx = tree_model.pivot_c
        pivot_val = tree_model.pivot[0]
        left_child_node = get_root_node(tree_model.left)
        right_child_node = get_root_node(tree_model.right)
        return Node('pconc',pivot_idx, pivot_val, True, tree_model.lm_l, left_child_node, tree_model.lm_r, right_child_node)
    else:
        pivot_idx, pivot_val = tree_model.pivot
        left_child_node = get_root_node(tree_model.left)
        right_child_node = get_root_node(tree_model.right)
        return Node(tree_model.node,pivot_idx, pivot_val, False, tree_model.lm_l, left_child_node, tree_model.lm_r,right_child_node)
