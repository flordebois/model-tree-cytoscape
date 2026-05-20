import numpy as np
import pandas as pd
from typing import List, Tuple
from datetime import datetime
import matplotlib.pyplot as plt

from viz_tree.nodes import BaseNode, LeafNode, InternalNode, CombinedLinNode, CollapsedNode
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
        nodes = [self.root_node] + self.root_node.get_all_children()
        return nodes

    def collect_edges(self) -> List[Tuple[BaseNode, BaseNode]]:
        edges = []
        for node in self.nodes:
            for child in node.get_children():
                edges.append((node, child))
        return edges

    def get_n_leafs(self) -> int:
        count = 0
        for node in self.nodes:
            if isinstance(node, LeafNode):
                count += 1
        return count

    def collapse(self, parent_node: InternalNode):
        for child in parent_node.get_all_children():
            if isinstance(child, CollapsedNode):
                self.expand(child)

        collapsed_node = CollapsedNode(parent_node)

        nodes_to_remove = collapsed_node.child_nodes
        for node in nodes_to_remove:
            self.nodes.remove(node)

        for node in nodes_to_remove:
            for child in node.get_children():
                self.edges.remove((node, child))
        for child in parent_node.get_children():
            self.edges.remove((parent_node, child))

        parent_node.left_child_node = collapsed_node
        parent_node.right_child_node = None
        self.nodes.append(collapsed_node)
        self.edges.append((parent_node, collapsed_node))

    def expand(self, node: CollapsedNode):
        parent_node = node.parent
        self.nodes.remove(node)
        self.edges.remove((parent_node, node))
        parent_node.left_child_node = node.parent_left_child_node
        parent_node.right_child_node = node.parent_right_child_node

        nodes_to_add = node.child_nodes
        for child in nodes_to_add:
            self.nodes.append(child)

        for node in nodes_to_add:
            for child in node.get_children():
                self.edges.append((node, child))
        for child in parent_node.get_children():
            self.edges.append((parent_node, child))

    def expand_all_nodes(self):
        nodes_to_expand = []
        for node in self.nodes:
            if isinstance(node, CollapsedNode):
                nodes_to_expand.append(node)
        for node in nodes_to_expand:
            self.expand(node)

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
        def internal_and_leaf_from_dict(node_dict):
            if node_dict["type"] == "leaf":
                node = LeafNode(
                    np.array(node_dict["indices"]),
                    np.array(node_dict["y_res"]),
                    np.array(node_dict["coefficients"]),
                    node_dict["intercept"],
                )

            else: # type in ["lin", "pcon", "plin", "blin", "pconc"]:
                node = InternalNode(
                    node_dict["type"],
                    np.array(node_dict["indices"]),
                    np.array(node_dict["y_res"]),
                    node_dict["pivot_idx"],
                    node_dict["pivot_value"],
                    tuple(node_dict["left_lin_model"]) if node_dict["left_lin_model"] else None,
                    None,  # temporary
                    tuple(node_dict["right_lin_model"]) if node_dict["right_lin_model"] else None,
                    None,  # temporary
                )

            return node

        id_to_node = {}
        for node_dict in viz_dict["nodes"]:
            if node_dict["type"] == "combined_lin":
                node = CombinedLinNode.__new__(CombinedLinNode)
                node.pivot_indices = node_dict["pivot_indices"]
                node.lin_coefficients = node_dict["lin_coefficients"]
                node.intercept = node_dict["intercept"]
                node.type = "combined_lin"
                node.indices = node_dict["indices"]
                node.y_res = np.array(node_dict["y_res"])
                node.child = None # temporary

            elif node_dict["type"] == "collapsed":
                node = CollapsedNode.__new__(CollapsedNode)
                node.type = "collapsed"
                node.indices = node_dict["indices"]
                node.y_res = np.array(node_dict["y_res"])
                node.n_nodes = node_dict["n_nodes"]
                node.parent = None  # temporary

                # - rebuild children -
                id_to_node2 = {}
                for node_dict2 in node_dict["child_nodes"]:
                    node2 = internal_and_leaf_from_dict(node_dict2)
                    node2.set_id(node_dict2["id"])
                    id_to_node2[node2.id] = node2

                # - reconnect children -
                for node_dict2 in node_dict["child_nodes"]:
                    node2 = id_to_node2[node_dict2["id"]]
                    if isinstance(node2, InternalNode):
                        node2.left_child_node = id_to_node2[node_dict2["left_child"]]
                        if node_dict2["right_child"] is not None:
                            node2.right_child_node = id_to_node2[node_dict2["right_child"]]

                node.child_nodes = list(id_to_node2.values())
                node.parent_left_child_node = id_to_node2[node_dict["parent_left_child_node"]]
                node.parent_right_child_node = id_to_node2[node_dict["parent_right_child_node"]] if node_dict["parent_right_child_node"] is not None else None

            else: # type is "leaf" or in ["lin", "pcon", "plin", "blin", "pconc"]
                node = internal_and_leaf_from_dict(node_dict)

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

            elif isinstance(node, CollapsedNode):
                node.parent = id_to_node[node_dict["parent"]]

        obj.nodes = list(id_to_node.values())

        # --- edges ---
        obj.edges = [(id_to_node[parent_node], id_to_node[child_node])
                     for parent_node, child_node in viz_dict["edges"]]

        # --- root ---
        obj.root_node = id_to_node[viz_dict["root_id"]]

        return obj