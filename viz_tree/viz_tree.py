import numpy as np
import pandas as pd
from typing import List, Tuple
import os
from datetime import datetime
from viz_tree.nodes import NODE_FONT_NAME
from viz_tree.nodes import BaseNode, LeafNode, InternalNode, CombinedLinNode
from viz_tree.build_viz_tree import build_viz_tree_from_pilot

class VizTree:
    root_node: BaseNode
    nodes: List[BaseNode]
    edges: List[Tuple[InternalNode, BaseNode]]
    X_train: np.ndarray
    y_train: np.ndarray
    feature_names: List[str]
    target_name: str
    rankdir: str
    output_directory: str
    tree_id: str

    def __init__(self, pilot_tree, X_train, y_train, feature_names= None, target_name= None,
                 rankdir='TD', output_directory: str = None, tree_id: str = None):

        self.feature_names = feature_names
        self.target_name = target_name
        self.rankdir = rankdir
        self.output_directory = output_directory
        if tree_id is None:
            self.tree_id = datetime.now().strftime('%d-%m-%y_%H-%M-%S')
        else:
            self.tree_id = tree_id

        if isinstance(X_train, pd.core.frame.DataFrame):
            self.X_train = np.array(X_train)
        else:
            self.X_train = X_train

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

    def get_dot(self, combine_lin=False, use_predsplot=False, use_regplot=False, highlight_x=None):
        if combine_lin and use_regplot:
            raise ValueError("You can't combine linear nodes when using regression plots")

        # Find nodes and edges to highlight along path for x
        highlight_nodes = []
        highlight_edges = []
        if highlight_x is not None:
            highlight_nodes.append(self.root_node)
            node_in_path = self.root_node
            while node_in_path.type != "leaf":
                parent_node_in_path = node_in_path
                if node_in_path.type == "lin":
                    node_in_path = node_in_path.left_child_node
                else:
                    if highlight_x[node_in_path.pivot_idx] > node_in_path.pivot_value:
                        node_in_path = node_in_path.right_child_node
                    else:
                        node_in_path = node_in_path.left_child_node
                highlight_nodes.append(node_in_path)
                highlight_edges.append((parent_node_in_path, node_in_path))

        # Combine multiple linear into one
        if combine_lin:
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

        if use_predsplot:
            directory_map = os.path.join(self.output_directory, "predsplots")
            os.makedirs(directory_map, exist_ok=True)
            directory_predsplot_map = os.path.join(directory_map, f"tree_id_{self.tree_id}")
        if use_regplot:
            directory_map2 = os.path.join(self.output_directory, "regplots")
            os.makedirs(directory_map2, exist_ok=True)
            directory_regplot_map = os.path.join(directory_map2, f"tree_id_{self.tree_id}")

        # Get .dot for each node
        nodes_dot = []
        for i, node in enumerate(nodes):
            highlight = node in highlight_nodes

            if isinstance(node, LeafNode) and use_predsplot:
                dot = node.get_dot_predsplot(i, directory_predsplot_map, highlight_x=highlight_x if highlight else None)
            elif isinstance(node, InternalNode) and use_regplot:
                dot = node.get_dot_regplot(i, directory_regplot_map, highlight_x=highlight_x if highlight else None)
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
                        linear_label += f'{np.round(parent_node.lin_coefficients[pivot_idx], 2)}X<SUB><FONT POINT-SIZE="9">{pivot_idx}</FONT></SUB> + '
                    label = (f'<table border="0"><tr><td border="0">'
                             f'{linear_label}{np.round(parent_node.intercept, 2)}'
                             f'</td></tr></table>')
                else:
                    if parent_node.left_child_node == child_node:
                        linear_model = parent_node.left_lin_model
                    else:
                        linear_model = parent_node.right_lin_model
                    if parent_node.type == "pcon" or parent_node.type == "pconc":
                        label = (f'<table border="0"><tr><td border="0">'
                                 f'{np.round(linear_model[1], 2)}'
                                 f'</td></tr></table>')
                    else:
                        label = (f'<table border="0"><tr><td border="0">'
                                 f'{np.round(linear_model[0], 2)}X'
                                 f'<SUB><FONT POINT-SIZE="9">{parent_node.pivot_idx}</FONT></SUB>'
                                 f' + {np.round(linear_model[1], 2)}'
                                 f'</td></tr></table>')

                edges_dot.append(f'node{parent_node.node_id} -> node{child_node.node_id}'
                                 f'[penwidth=3 fontname={NODE_FONT_NAME}, label=<{label}>]')
            else:
                edges_dot.append(f"node{parent_node.node_id} -> node{child_node.node_id}")

        # Combine .dot of nodes and edges
        newline = "\n\t"
        dot = f"""
                digraph G {{
                    splines=line;
                    rankdir={self.rankdir};

                    {newline.join(nodes_dot)}
                    {newline.join(edges_dot)}
                }}
            """

        return dot
