import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from shap.plots import colors
from datetime import datetime
import os
from subprocess import run

from viz_tree.viz_tree import VizTree
from nodes.collapsed_node import CollapsedNode
from nodes.internal_node import InternalNode, LinearNode
from nodes.split_node import PconNode, PconcNode, SplitNode
from nodes.leaf_node import LeafNode
from plots.beeswarm import beeswarm
import matplotlib.pyplot as plt

# def get_contributions(viz_tree: VizTree):
#     X = viz_tree.X_train
#     all_contributions = np.zeros_like(X, dtype=float)
#     for node in viz_tree.nodes:
#         if isinstance(node, CollapsedNode):
#             raise ValueError("CollapsedNode cannot be used when calculating contributions")
#         if isinstance(node, InternalNode):
#             pivot = node.pivot_idx
#             predictions_left = node.left_model.predict(X[node.left_child_node.indices, pivot])
#             y_hat_node = np.mean(predictions_left) * np.sum(node.left_child_node.indices)
#             if node.right_child_node is not None:
#                 predictions_right = node.right_lin_model[1] + node.right_lin_model[0] * X[node.right_child_node.indices, pivot]
#                 y_hat_node += np.mean(predictions_right) * np.sum(node.right_child_node.indices)
#
#             y_hat_node = y_hat_node/np.sum(node.indices)
#             all_contributions[node.left_child_node.indices, pivot] += predictions_left - y_hat_node
#             if node.right_child_node is not None:
#                 all_contributions[node.right_child_node.indices, pivot] += predictions_right - y_hat_node
#     return all_contributions

def get_split_and_lin_contributions(viz_tree: VizTree):
    X = viz_tree.X_train
    split_contributions = np.zeros_like(X, dtype=float)
    lin_contributions = np.zeros_like(X, dtype=float)
    for node in viz_tree.nodes:
        if isinstance(node, CollapsedNode):
            raise ValueError("CollapsedNode cannot be used when calculating contributions")
        if isinstance(node, InternalNode):
            pivot = node.pivot_idx
            if isinstance(node, LinearNode):
                linear_predictions = node.linear_model.predict(X[node.child.indices, pivot])
                lin_contributions[node.child.indices, pivot] += linear_predictions - np.mean(linear_predictions)
                continue
            if isinstance(node, PconcNode) or isinstance(node, PconNode):
                avg_pred_left = node.left_model.predict([0])[0]
                avg_pred_right = node.right_model.predict([0])[0]
            elif isinstance(node, SplitNode): #blin and plin
                linear_predictions_left = node.left_model.predict(X[node.left_child.indices, pivot])
                linear_predictions_right = node.right_model.predict(X[node.right_child.indices, pivot])
                avg_pred_left = np.mean(linear_predictions_left)
                avg_pred_right = np.mean(linear_predictions_right)
                lin_contributions[node.left_child.indices, pivot] += linear_predictions_left - avg_pred_left
                lin_contributions[node.right_child.indices, pivot] += linear_predictions_right - avg_pred_right
            else:
                raise ValueError(f"Node of class {node.__class__.__name__} is not implemented to calculate contributions")

            y_hat_node = (avg_pred_left*np.sum(node.left_child.indices) + avg_pred_right*np.sum(node.right_child.indices))/np.sum(node.indices)
            split_contributions[node.left_child.indices, pivot] += avg_pred_left - y_hat_node
            split_contributions[node.right_child.indices, pivot] += avg_pred_right - y_hat_node
    return split_contributions, lin_contributions

def get_split_and_lin_contributions_test(viz_tree: VizTree, x_test):
    split_contributions = np.zeros_like(x_test, dtype=float)
    lin_contributions = np.zeros_like(x_test, dtype=float)

    X = viz_tree.X_train
    node = viz_tree.root_node
    while isinstance(node, LeafNode):
        # Calculate contributions
        if isinstance(node, CollapsedNode):
            raise ValueError("CollapsedNode cannot be used when calculating contributions")
        if isinstance(node, InternalNode):
            pivot = node.pivot_idx
            if isinstance(node, LinearNode):
                linear_predictions = node.linear_model.predict(X[node.child.indices, pivot])
                linear_prediction_x_test = node.linear_model.predict([x_test[pivot]])
                lin_contributions[pivot] += linear_prediction_x_test - np.mean(linear_predictions)
            else:
                if isinstance(node, PconcNode) or isinstance(node, PconNode):
                    avg_pred_left = node.left_model.predict([0])[0]
                    avg_pred_right = node.right_model.predict([0])[0]
                elif isinstance(node, SplitNode):  # blin and plin
                    linear_predictions_left = node.left_model.predict(X[node.left_child.indices, pivot])
                    linear_predictions_right = node.right_model.predict(X[node.right_child.indices, pivot])
                    avg_pred_left = np.mean(linear_predictions_left)
                    avg_pred_right = np.mean(linear_predictions_right)
                    linear_prediction_x_test_left = node.left_model.predict([x_test[pivot]])
                    linear_prediction_x_test_right = node.right_model.predict([x_test[pivot]])
                else:
                    raise ValueError(f"Node of class {node.__class__.__name__} is not implemented to calculate contributions")

                y_hat_node = (avg_pred_left*np.sum(node.left_child.indices) + avg_pred_right*np.sum(node.right_child.indices))/np.sum(node.indices)

        # Find next node in path
        if isinstance(node, LinearNode):
            node = node.child
        elif isinstance(node, PconcNode):
            if x_test[node.pivot_idx] in node.pivot_value:
                node = node.left_child
                split_contributions[pivot] += avg_pred_left - y_hat_node
            else:
                node = node.right_child
                split_contributions[pivot] += avg_pred_right - y_hat_node
        else:
            if x_test[node.pivot_idx] > node.pivot_value:
                split_contributions[pivot] += avg_pred_right - y_hat_node
                if not isinstance(node, PconNode):
                    lin_contributions[pivot] += linear_prediction_x_test_right - avg_pred_right
                node = node.right_child
            else:
                split_contributions[pivot] += avg_pred_left - y_hat_node
                if not isinstance(node, PconNode):
                    lin_contributions[pivot] += linear_prediction_x_test_left - avg_pred_left
                node = node.left_child

    return split_contributions, lin_contributions

def scatter_contributions(df_X, df_contributions, feature, color_feature = None):
    if color_feature is None:
        corr = df_contributions.corr()
        color_feature = corr[feature].nlargest(2).index[1]

    x_color = df_X[color_feature].to_numpy()
    median = np.nanmedian(x_color)
    mad = 1.4826 * np.nanmedian(np.abs(x_color - median))
    vmin = median - 3 * mad
    vmax = median + 3 * mad


    fig, ax = plt.subplots(figsize=(8, 6))

    # scatter
    sc = ax.scatter(
        df_X[feature],
        df_contributions[feature],
        c=df_X[color_feature].clip(vmin, vmax),
        cmap=colors.red_blue,
        s=10,
        marker='.',
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(f"{feature} contribution")
    ax.set_xlabel(feature)
    ax.set_ylabel("Contribution")

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(color_feature)

    # histogram inset at bottom
    ax_hist = inset_axes(
        ax,
        width="100%",
        height="18%",
        loc="lower left",
        bbox_to_anchor=(0, 0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )

    ax_hist.hist(
        df_X[feature],
        bins=50,
        color="gray",
        alpha=0.3
    )

    ax_hist.set_yticks([])
    ax_hist.set_xticks([])
    ax_hist.patch.set_alpha(0)

    for spine in ax_hist.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.show()

def beeswarm_wrap(df_contributions, X, y_hat, directory):
    id_results = datetime.now().strftime('%d-%m-%y_%H-%M-%S')
    file_directory = os.path.join(directory, f"beeswarm_{id_results}.pdf")
    feature_names = df_contributions.columns.tolist()
    beeswarm(np.array(df_contributions),
             X,
             y_hat,
             n_max=10,
             fig_size=(10, 6),
             truncate_total_pred=True,
             variable_tick_width=True,
             file_directory=file_directory,
             highlight_x=None,
             staircase=False,
             feature_names=feature_names,
             )
    run(["open", "-a", "Preview", file_directory])