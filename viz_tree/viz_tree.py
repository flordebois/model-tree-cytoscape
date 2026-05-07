import numpy as np
import pandas as pd
from typing import List, Tuple
from datetime import datetime
import matplotlib.pyplot as plt

from viz_tree.nodes import BaseNode, LeafNode, InternalNode, CombinedLinNode
from viz_tree.build_viz_tree import build_viz_tree_from_pilot

class VizTree:
    root_node: BaseNode
    nodes: List[BaseNode]
    edges: List[Tuple[BaseNode, BaseNode]]
    X_train: np.ndarray
    feature_colors: List
    y_train: np.ndarray
    output_directory: str
    tree_id: str

    def __init__(self, pilot_tree, X_train, y_train, output_directory: str = None, tree_id: str = None):
        self.output_directory = output_directory # TODO: is not used at the moment!
        if tree_id is None:
            self.tree_id = datetime.now().strftime('%d-%m-%y_%H-%M-%S')
        else:
            self.tree_id = tree_id

        if isinstance(X_train, pd.core.frame.DataFrame):
            self.X_train = np.array(X_train)
        else:
            self.X_train = X_train

        n_features = self.X_train.shape[1]
        cmap = plt.cm.get_cmap('tab20', n_features)
        self.feature_colors = [cmap(i) for i in range(n_features)]

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
        self.nodes = self.collect_nodes()
        self.edges = self.collect_edges()
        for i, node in enumerate(self.nodes):
            node.set_id(i)

    def collect_nodes(self) -> List[BaseNode]:
        def traverse_tree(node):
            if node is None:
                return []
            return [node] + sum([traverse_tree(c) for c in node.get_children()], [])

        return traverse_tree(self.root_node)

    def collect_edges(self) -> List[Tuple[BaseNode, BaseNode]]:
        edges = []
        for node in self.nodes:
            for child in node.get_children():
                edges.append((node, child))
        return edges

    def to_dict(self):
        return {
            "tree_id": self.tree_id,
            "X_train": self.X_train.tolist(),
            "y_train": self.y_train.tolist(),
            "feature_colors": list(self.feature_colors),
            "output_directory": self.output_directory,

            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [(parent_node.id, child_node.id) for parent_node, child_node in self.edges],
            "root_id": self.root_node.id
        }

    @classmethod
    def from_dict(cls, viz_dict):
        obj = cls.__new__(cls)  # bypass __init__

        obj.tree_id = viz_dict["tree_id"]
        obj.X_train = np.array(viz_dict["X_train"])
        obj.y_train = np.array(viz_dict["y_train"])
        obj.feature_colors = viz_dict["feature_colors"]
        obj.output_directory = viz_dict["output_directory"]

        # --- rebuild nodes ---
        id_to_node = {}

        for node_dict in viz_dict["nodes"]:
            type = node_dict["type"]

            if type == "leaf":
                node = LeafNode(
                    np.array(node_dict["indices"]),
                    np.array(node_dict["y_res"]),
                    np.array(node_dict["coefficients"]),
                    node_dict["intercept"],
                )

            elif type == "combined_lin":
                node = CombinedLinNode.__new__(CombinedLinNode)
                node.pivot_indices = node_dict["pivot_indices"]
                node.lin_coefficients = node_dict["lin_coefficients"]
                node.intercept = node_dict["intercept"]
                node.type = "combined_lin"
                node.y_res = np.array(node_dict["y_res"])
                node.child = None

            else: # type in ["lin", "pcon", "plin", "blin", "pconc"]:
                node = InternalNode(
                    type,
                    np.array(node_dict["indices"]),
                    np.array(node_dict["y_res"]),
                    node_dict["pivot_idx"],
                    node_dict["pivot_value"],
                    tuple(node_dict["left_lin_model"]) if node_dict["left_lin_model"] else None,
                    None,  # temporary
                    tuple(node_dict["right_lin_model"]) if node_dict["right_lin_model"] else None,
                    None, # temporary
                )

            node.set_id(node_dict["id"])
            id_to_node[node.id] = node

        # --- reconnect children ---
        for node_dict in viz_dict["nodes"]:
            node = id_to_node[node_dict["id"]]

            if isinstance(node, InternalNode):
                node.left_child_node = id_to_node[node_dict["left_child"]]
                if node_dict["right_child"] is not None:
                    node.right_child_node = id_to_node[node_dict["right_child"]]

            elif isinstance(node, CombinedLinNode):
                node.child = id_to_node[node_dict["child"]]

        obj.nodes = list(id_to_node.values())

        # --- edges ---
        obj.edges = [(id_to_node[parent_node], id_to_node[child_node])
                     for parent_node, child_node in viz_dict["edges"]]

        # --- root ---
        obj.root_node = id_to_node[viz_dict["root_id"]]

        return obj