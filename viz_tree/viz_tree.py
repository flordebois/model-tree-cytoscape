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
        for i, node in enumerate(self.nodes):
            node.set_id(i)

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
