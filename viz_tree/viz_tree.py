import numpy as np
from typing import List, Tuple
from datetime import datetime

from nodes.base_node import BaseNode
from nodes.leaf_node import LeafNode
from nodes.internal_node import InternalNode, LinearNode
from nodes.collapsed_node import CollapsedNode
from nodes.none_node import NoneNode
from nodes.node_model import LinearNodeModel, ConstantNodeModel
from nodes.split_node import SplitNode


class VizTree:
    root_node: BaseNode
    nodes: List[BaseNode]
    edges: List[Tuple[BaseNode, BaseNode]]
    X_train: np.ndarray
    y_train: np.ndarray
    y_hat: np.ndarray
    tree_id: str

    def __init__(self, root, X_train, y_train, y_hat, tree_id: str = None):
        if tree_id is None:
            self.tree_id = datetime.now().strftime('%d-%m-%y_%H-%M-%S')
        else:
            self.tree_id = tree_id

        self.X_train = X_train
        self.y_train = y_train
        self.y_hat = y_hat

        self.root_node = root
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

    def get_depth(self, node=None) -> int:
        if node is None:
            node = self.root_node
        elif isinstance(node, LeafNode):
            return 0
        elif isinstance(node, NoneNode):
            return -1
        elif isinstance(node, CollapsedNode):
            return self.get_depth(node.parent)-1
        return max([self.get_depth(node) for node in node.get_children()]) + 1


    def get_n_leafs(self) -> int:
        count = 0
        for node in self.nodes:
            if isinstance(node, LeafNode):
                count += 1
        return count

    def collapse(self, parent_node: InternalNode) -> int:
        for child in parent_node.get_all_children():
            if isinstance(child, CollapsedNode):
                self.expand(child)

        collapsed_node = CollapsedNode(parent_node)
        parent_node.set_children([collapsed_node, NoneNode()])

        self.nodes = self.collect_nodes()
        self.edges = self.collect_edges()

        return collapsed_node.n_nodes

    def expand(self, collapsed_node: CollapsedNode):
        for parent_node in self.nodes:
            if parent_node.id == collapsed_node.parent_id:
                break
        parent_node.set_children(collapsed_node.parent.get_children())

        self.nodes = self.collect_nodes()
        self.edges = self.collect_edges()

    def expand_all_nodes(self):
        nodes_to_expand = []
        for node in self.nodes:
            if isinstance(node, CollapsedNode):
                nodes_to_expand.append(node)
        for node in nodes_to_expand:
            self.expand(node)

    def _get_path_to_node(self, to_find_node: BaseNode, current_node=None) -> List[BaseNode]:
        if current_node is None:
            current_node = self.root_node

        if current_node is to_find_node:
            return [current_node]

        if isinstance(current_node, LeafNode):
            return []

        for child in current_node.get_children():
            path = self._get_path_to_node(to_find_node, child)
            if path:
                return [current_node] + path
        return []

    def prune(self, prune_node: InternalNode):
        self.expand_all_nodes()
        node_path = self._get_path_to_node(prune_node)
        model = LinearNodeModel(coefficients = np.zeros(self.X_train.shape[1]), intercept = 0)

        for i in range(len(node_path)-1):
            node = node_path[i]
            if isinstance(node, LinearNode):
                model.add_model(node.linear_model)
            elif isinstance(node, SplitNode):
                next_node = node_path[i+1]
                if next_node is node.left_child:
                    model.add_model(node.left_model)
                else:
                    model.add_model(node.right_model)
            else:
                raise NotImplementedError

        if np.sum(model.coefficients) == 0:
            model = ConstantNodeModel(model.intercept)

        new_leaf_node = LeafNode(
            indices=prune_node.indices,
            y_res=prune_node.y_res,
            rss=prune_node.rss,
            node_model=model,
        )
        new_leaf_node.set_id(prune_node.id)

        if isinstance(node, LinearNode):
            node.child = new_leaf_node
        elif isinstance(node, SplitNode):
            if prune_node is node.left_child:
                node.left_child = new_leaf_node
            else:
                node.right_child = new_leaf_node
        else:
            raise NotImplementedError

        self.nodes = self.collect_nodes()
        self.edges = self.collect_edges()



    def to_dict(self):
        return {
            "tree_id": self.tree_id,
            "X_train": self.X_train.tolist(),
            "y_train": self.y_train.tolist(),
            "y_hat": self.y_hat.tolist(),

            "root_node_class": self.root_node.__class__.__name__,
            "root_node_dict": self.root_node.to_dict(),
        }

    @classmethod
    def from_dict(cls, viz_dict):
        root_child_class = BaseNode.class_registry[viz_dict["root_node_class"]]

        obj = cls.__new__(cls)
        obj.tree_id = viz_dict["tree_id"]
        obj.X_train = np.array(viz_dict["X_train"])
        obj.y_train = np.array(viz_dict["y_train"])
        obj.y_hat = np.array(viz_dict["y_hat"])

        obj.root_node = root_child_class.from_dict(viz_dict["root_node_dict"])
        obj.nodes = obj.collect_nodes()
        obj.edges = obj.collect_edges()

        return obj
