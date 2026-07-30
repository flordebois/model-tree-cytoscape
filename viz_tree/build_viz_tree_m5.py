import numpy as np
from m5py.main import ConstantLeafModel, LinRegLeafModel

from nodes.base_node import BaseNode
from nodes.leaf_node import LeafNode
from nodes.split_node import SplitNode
from nodes.node_model import LinearNodeModel, NoneNodeModel, ConstantNodeModel

def build_viz_tree_from_m5(m5_model, X_train, y_train) -> BaseNode:
    # builds a viz tree from a fitted m5py model (M5Base / M5Prime).
    tree = m5_model.tree_
    node_models = m5_model.node_models
    root_indices = np.ones(X_train.shape[0], dtype=bool)

    return _build_viz_tree_from_m5(tree, node_models, X_train, y_train, root_indices, node_id=0)

def _build_viz_tree_from_m5(tree, node_models, X_train, y_train, current_indices, node_id) -> BaseNode:

    current_y_res = y_train[current_indices]
    n_features = X_train.shape[1]

    left_id = tree.children_left[node_id]
    right_id = tree.children_right[node_id]

    if left_id == -1:  # sklearn's TREE_LEAF sentinel; children_left/right are both -1 at leaves
        model = node_models[node_id]

        if isinstance(model, ConstantLeafModel):
            rss = model.error  # note: this is an RMSE, not a raw RSS like PILOT's Rt - rescale/rename if needed
            node_model = ConstantNodeModel(float(tree.value[node_id].ravel()[0]))

        elif isinstance(model, LinRegLeafModel):
            coefficients = np.zeros(n_features)
            coefficients[model.features] = model.model.coef_
            intercept = model.model.intercept_
            rss = model.error  # same caveat as above
            node_model = LinearNodeModel(coefficients, intercept)

        else:
            raise ValueError(f"Unexpected leaf model type: {type(model)}")

        return LeafNode(
            indices=current_indices,
            y_res=current_y_res,
            rss=rss,
            node_model=node_model,
        )

    else:
        pivot_idx = tree.feature[node_id]
        pivot_value = tree.threshold[node_id]

        left_mask = X_train[current_indices, pivot_idx] <= pivot_value
        right_mask = ~left_mask

        left_indices, right_indices = current_indices.copy(), current_indices.copy()
        left_indices[current_indices] = left_mask
        right_indices[current_indices] = right_mask

        left_child = _build_viz_tree_from_m5(tree, node_models, X_train, y_train, left_indices, node_id=left_id)
        right_child = _build_viz_tree_from_m5(tree, node_models, X_train, y_train, right_indices, node_id=right_id)

        rss = tree.impurity[node_id] * tree.n_node_samples[node_id]  # sklearn stores impurity as MSE; * n -> RSS

        return SplitNode(
            indices=current_indices,
            y_res=current_y_res,
            rss=rss,
            pivot_idx=pivot_idx,
            pivot_value=pivot_value,
            left_child=left_child,
            right_child=right_child,
            left_model=NoneNodeModel(),
            right_model=NoneNodeModel(),
        )
