import numpy as np
import pandas as pd
from typing import List, Tuple
import os
from datetime import datetime
import matplotlib.pyplot as plt

from viz_tree.nodes import NODE_FONT_NAME
from viz_tree.nodes import BaseNode, LeafNode, InternalNode, CombinedLinNode
from viz_tree.build_viz_tree import build_viz_tree_from_pilot
from viz_tree.dot_settings import DotSettings

class VizTree:
    root_node: BaseNode
    nodes: List[BaseNode]
    edges: List[Tuple[InternalNode, BaseNode]]
    X_train: np.ndarray
    feature_colors: List
    y_train: np.ndarray
    output_directory: str
    tree_id: str

    def __init__(self, pilot_tree, X_train, y_train, output_directory: str = None, tree_id: str = None):
        self.output_directory = output_directory
        if tree_id is None:
            self.tree_id = datetime.now().strftime('%d-%m-%y_%H-%M-%S')
        else:
            self.tree_id = tree_id

        if isinstance(X_train, pd.core.frame.DataFrame):
            self.X_train = np.array(X_train)
        else:
            self.X_train = X_train

        cmap = plt.cm.tab10.colors
        # TODO: make robust for each dataset
        self.feature_colors = [cmap[3], cmap[1], cmap[9], cmap[2], "grey", "grey", cmap[5]]
        # cmap = plt.cm.tab10
        # self.feature_colors = cmap(np.linspace(0, 1, X_train.shape[1]))

        if isinstance(y_train, pd.core.frame.DataFrame):
            self.y_train = np.array(y_train)
        else:
            self.y_train = y_train

        self.root_node = build_viz_tree_from_pilot(
            pilot_tree,
            X_train,
            current_indices=np.ones(len(X_train), dtype=bool),
            current_y_res=y_train,
            accumulated_coefficients=np.zeros(X_train.shape[1]),
            accumulated_intercept=0.0
        )
        self.nodes = self._collect_nodes()
        self.edges = self._collect_edges()

    def _collect_nodes(self) -> List[BaseNode]:
        def traverse_tree(node):
            if node is None:
                return []
            return [node] + sum([traverse_tree(c) for c in node.get_children()], [])

        return traverse_tree(self.root_node)

    def _collect_edges(self) -> List[Tuple[InternalNode, BaseNode]]:
        edges = []
        for node in self.nodes:
            for child in node.get_children():
                edges.append((node, child))
        return edges

    def get_dot(self, dot_set: DotSettings):
        if dot_set.combine_lin and dot_set.use_regplot:
            raise ValueError("You can't combine linear nodes when using regression plots")

        # Find nodes and edges to highlight along path for x
        highlight_nodes = []
        highlight_edges = []
        if dot_set.highlight_x is not None:
            highlight_nodes.append(self.root_node)
            node_in_path = self.root_node
            while node_in_path.type != "leaf":
                parent_node_in_path = node_in_path
                if node_in_path.type == "lin":
                    node_in_path = node_in_path.left_child_node
                else:
                    if dot_set.highlight_x[node_in_path.pivot_idx] > node_in_path.pivot_value:
                        node_in_path = node_in_path.right_child_node
                    else:
                        node_in_path = node_in_path.left_child_node
                highlight_nodes.append(node_in_path)
                highlight_edges.append((parent_node_in_path, node_in_path))

        # Combine multiple linear into one
        if dot_set.combine_lin:
            nodes = []
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
                    for lin_node in combine_nodes:
                        new_node_mapping[lin_node] = combined_lin_node
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
                    new_edge = (new_parent_node, new_child_node)
                    edges.append(new_edge)
                    if edge in highlight_edges:
                        highlight_edges.append(new_edge)

        # No combining of linear nodes
        else:
            nodes = self.nodes
            edges = self.edges

        if dot_set.use_predsplot:
            directory_map = os.path.join(self.output_directory, "predsplots")
            os.makedirs(directory_map, exist_ok=True)
            directory_predsplot_map = os.path.join(directory_map, f"tree_id_{self.tree_id}")
        if dot_set.use_regplot:
            directory_map2 = os.path.join(self.output_directory, "regplots")
            os.makedirs(directory_map2, exist_ok=True)
            directory_regplot_map = os.path.join(directory_map2, f"tree_id_{self.tree_id}")

        # Get .dot for each node
        nodes_dot = []
        for i, node in enumerate(nodes):
            highlight = node in highlight_nodes

            if isinstance(node, LeafNode):
                if dot_set.use_predsplot:
                    dot = node.get_dot_predsplot(i, directory_predsplot_map, dot_set, self.feature_colors, highlight)
                else:
                    dot = node.get_dot(i, print_model = dot_set.print_model)
            elif isinstance(node, InternalNode) and dot_set.use_regplot:
                dot = node.get_dot_regplot(i, directory_regplot_map, dot_set, self.feature_colors, highlight)
            else:
                dot = node.get_dot(i)
                if highlight:
                    dot = dot[:-1] + ", penwidth=3]"
            nodes_dot.append(dot)

        # Get .dot for each edge
        edges_dot = []
        for edge in edges:
            parent_node, child_node = edge
            if edge in highlight_edges:
                if isinstance(parent_node, CombinedLinNode):
                    linear_label = ""
                    for pivot_idx in parent_node.lin_coefficients.keys():
                        linear_label += f'{parent_node.lin_coefficients[pivot_idx]:.3g}X<SUB><FONT POINT-SIZE="9">{pivot_idx}</FONT></SUB> + '
                    label = (f'<table border="0"><tr><td border="0">'
                             f'{linear_label}{parent_node.intercept:.3g}'
                             f'</td></tr></table>')
                else:
                    if parent_node.left_child_node == child_node:
                        linear_model = parent_node.left_lin_model
                    else:
                        linear_model = parent_node.right_lin_model
                    if parent_node.type == "pcon" or parent_node.type == "pconc":
                        label = (f'<table border="0"><tr><td border="0">'
                                 f'{linear_model[1]:.3g}'
                                 f'</td></tr></table>')
                    else:
                        label = (f'<table border="0"><tr><td border="0">'
                                 f'{linear_model[0]:.3g}X'
                                 f'<SUB><FONT POINT-SIZE="9">{parent_node.pivot_idx}</FONT></SUB>'
                                 f' + {linear_model[1]:.3g}'
                                 f'</td></tr></table>')

                edges_dot.append(f'node{parent_node.node_id} -> node{child_node.node_id}'
                                 f'[penwidth=3 fontname={NODE_FONT_NAME}, label=<{label}>]')
            else:
                edges_dot.append(f"node{parent_node.node_id} -> node{child_node.node_id}")
                                 #f" [penwidth={child_node.X.shape[0]/self.X_train.shape[0]*50}, arrowhead=none]")

        # Combine .dot of nodes and edges
        newline = "\n\t"
        dot = f"""
                digraph G {{
                    splines=line;
                    rankdir={dot_set.rankdir};

                    {newline.join(nodes_dot)}
                    {newline.join(edges_dot)}
                }}
            """

        return dot
