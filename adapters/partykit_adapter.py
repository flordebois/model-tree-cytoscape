import os
import json
import numpy as np
from adapters.adapter import BaseAdapter, register_adapter
from adapters.adapter_utils import split_by_categories, split_by_threshold
from nodes.base_node import BaseNode
from nodes.leaf_node import LeafNode
from nodes.split_node import SplitNode, SplitCNode
from nodes.node_model import LinearNodeModel, NoneNodeModel

@register_adapter('Partykit')
class PartyKitAdapter(BaseAdapter):
    @staticmethod
    def load_model(model_path):
        if not os.path.exists(model_path):
            raise ValueError(f"No such file path to load model: {model_path}")
        with open(model_path) as f:
            model = json.load(f)

        return model

    @staticmethod
    def build_root_node(X_train: np.ndarray, y_train: np.ndarray, model) -> BaseNode:
        root_indices = np.ones(X_train.shape[0], dtype=bool)
        node_list = model['nodes']
        feature_names = model['names'][1:]
        return PartyKitAdapter._build_recursive(node_list, feature_names, node_list[0], X_train, y_train, root_indices)

    @staticmethod
    def _build_recursive(node_list, feature_names, node, X_train, y_train, current_indices) -> BaseNode:
        current_y_res = y_train[current_indices]
        n_features = X_train.shape[1]

        if node["is_terminal"]:
            coefficients = np.array([
                node["coefficients"].get(name, 0.0)
                for name in feature_names
            ], dtype=float)

            node_model = LinearNodeModel(coefficients, node["coefficients"]['(Intercept)'])

            return LeafNode(
                indices=current_indices,
                y_res=current_y_res,
                rss=-1, #TODO
                node_model=node_model,
            )
        else:
            pivot_idx = feature_names.index(node["split_var"])
            left_node = node_list[node["kids"][0]]
            right_node = node_list[node["kids"][1]]
            if "breaks" in node:
                pivot_value = node["breaks"]
                left_indices, right_indices = split_by_threshold(X_train, current_indices, pivot_idx, pivot_value)

                left_child = PartyKitAdapter._build_recursive(
                    node_list, feature_names, left_node, X_train, y_train, left_indices)
                right_child = PartyKitAdapter._build_recursive(
                    node_list, feature_names, right_node, X_train, y_train, right_indices)

                return SplitNode(
                    indices=current_indices,
                    y_res=current_y_res,
                    rss=-1, #TODO
                    pivot_idx=pivot_idx,
                    pivot_value=pivot_value,
                    left_child=left_child,
                    right_child=right_child,
                    left_model=NoneNodeModel(),
                    right_model=NoneNodeModel(),
                )
            elif "index" in node:
                left_categories = [lvl for lvl, idx in zip(node["levels"], node["index"]) if idx == 1]
                left_indices, right_indices = split_by_categories(X_train, current_indices, pivot_idx, left_categories)

                left_child = PartyKitAdapter._build_recursive(
                    node_list, feature_names, left_node, X_train, y_train, left_indices)
                right_child = PartyKitAdapter._build_recursive(
                    node_list, feature_names, right_node, X_train, y_train, right_indices)
                return SplitCNode(
                    indices=current_indices,
                    y_res=current_y_res,
                    rss=-1, #TODO
                    pivot_idx=pivot_idx,
                    pivot_value=left_categories,
                    left_child=left_child,
                    right_child=right_child,
                    left_model=NoneNodeModel(),
                    right_model=NoneNodeModel(),
                )
            else:
                raise ValueError(f"Unknown node: {node}")