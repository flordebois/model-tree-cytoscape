from typing import List
import numpy as np
from nodes.base_node import BaseNode
from nodes.node_model import NodeModel, ConstantNodeModel, node_model_from_dict
from nodes.internal_node import InternalNode


@BaseNode.register
class SplitNode(InternalNode):
    """Base split node with a left and right child.

    Attributes:
        pivot_value: Threshold value the split compares against.
        left_child: Child node for samples on the left of the split.
        right_child: Child node for samples on the right of the split.
        left_model: Model applied to samples on the left of the split.
        right_model: Model applied to samples on the right of the split.
    """

    pivot_value: float
    left_child: BaseNode
    right_child_node: BaseNode
    left_model: NodeModel
    right_model: NodeModel

    def __init__(self, indices: np.ndarray, y_res: np.ndarray, rss: float,
                 pivot_idx: int, pivot_value: float,
                 left_child: BaseNode, right_child: BaseNode,
                 left_model: NodeModel, right_model: NodeModel):
        """
        Args:
            indices: Row indices of the samples belonging to this node.
            y_res: Residuals of the target variable at this node.
            rss: Residual sum of squares at this node.
            pivot_idx: Index of the feature the split is based on.
            pivot_value: Threshold value the split compares against.
            left_child: Child node for samples on the left of the split.
            right_child: Child node for samples on the right of the split.
            left_model: Model applied to samples on the left of the split.
            right_model: Model applied to samples on the right of the split.
        """
        super().__init__(indices, y_res, rss, pivot_idx)
        self.pivot_value = pivot_value
        self.left_child = left_child
        self.right_child = right_child
        self.left_model = left_model
        self.right_model = right_model

    def get_children(self) -> List[BaseNode]:
        """Returns this node's left and right children.

        Returns:
            List of [left_child, right_child].
        """
        return [self.left_child, self.right_child]

    def set_children(self, children: List[BaseNode]):
        """Replaces this node's left and right children.

        Args:
            children: List of [new_left_child, new_right_child].
        """
        self.left_child = children[0]
        self.right_child = children[1]

    def get_label(self) -> str:
        """Returns the full display label for this node.

        Returns:
            Label of the form "SPLIT\\nX<pivot_idx> > <pivot_value>".
        """
        # Default label for pure split nodes
        return f"SPLIT\nX{self.pivot_idx} > {self.pivot_value:.3g}"

    def to_dict(self) -> dict:
        """Serializes the node to a dictionary.

        Returns:
            Dictionary with base attributes plus pivot_idx, pivot_value,
            both children's class names and serialized forms, and both
            models' serialized forms.
        """
        node_dict = super().to_dict()
        node_dict.update({
            "pivot_idx": self.pivot_idx,
            "pivot_value": self.pivot_value,
            "left_child_class": self.left_child.__class__.__name__,
            "left_child_dict": self.left_child.to_dict(),
            "right_child_class": self.right_child.__class__.__name__,
            "right_child_dict": self.right_child.to_dict(),
            "left_model": self.left_model.to_dict(),
            "right_model": self.right_model.to_dict(),
        })
        return node_dict

    @classmethod
    def from_dict(cls, dic: dict) -> "SplitNode":
        """Reconstructs a SplitNode instance, including children and models.

        Args:
            dic: Dictionary previously produced by to_dict().

        Returns:
            A new SplitNode instance with children and side models restored.
        """
        left_child_class = cls.class_registry[dic["left_child_class"]]
        right_child_class = cls.class_registry[dic["right_child_class"]]

        obj = super().from_dict(dic)
        obj.pivot_idx = dic["pivot_idx"]
        obj.pivot_value = dic["pivot_value"]
        obj.left_child = left_child_class.from_dict(dic["left_child_dict"])
        obj.right_child = right_child_class.from_dict(dic["right_child_dict"])
        obj.left_model = node_model_from_dict(dic["left_model"])
        obj.right_model = node_model_from_dict(dic["right_model"])
        return obj

@BaseNode.register
class PlinNode(SplitNode):
    """PLIN split node: split with linear models on both sides."""

    def get_label(self) -> str:
        """Returns the full display label for this node.

        Returns:
            Label of the form "PLIN\\nX<pivot_idx> > <pivot_value>".
        """
        return f"PLIN\nX{self.pivot_idx} > {self.pivot_value:.3g}"

@BaseNode.register
class BlinNode(SplitNode):
    """BLIN split node: split with a shared/blended linear model."""

    def get_label(self) -> str:
        """Returns the full display label for this node.

        Returns:
            Label of the form "BLIN\\nX<pivot_idx> > <pivot_value>".
        """
        return f"BLIN\nX{self.pivot_idx} > {self.pivot_value:.3g}"

@BaseNode.register
class PconNode(SplitNode):
    """PCON split node: split with constant models on both sides."""

    def get_label(self) -> str:
        """Returns the full display label for this node.

        Returns:
            Label of the form "PCON\\nX<pivot_idx> > <pivot_value>".
        """
        return f"PCON\nX{self.pivot_idx} > {self.pivot_value:.3g}"

@BaseNode.register
class PconcNode(PconNode):
    """PCONC split node: categorical PCON variant.

    Splits samples on category membership (pivot_value holds the set
    of categories routed to the left child), using constant models on both sides.
    """

    def __init__(self, indices: np.ndarray, y_res: np.ndarray, rss: float, pivot_idx: int, pivot_value: float,
                 left_child: BaseNode, right_child: BaseNode,
                 left_model: ConstantNodeModel, right_model: ConstantNodeModel):
        """
        Args:
            indices: Row indices of the samples belonging to this node.
            y_res: Residuals of the target variable at this node.
            rss: Residual sum of squares at this node.
            pivot_idx: Index of the categorical feature the split is based on.
            pivot_value: Category or set of categories defining the split.
            left_child: Child node for samples on the left of the split.
            right_child: Child node for samples on the right of the split.
            left_model: Constant model applied to the left of the split.
            right_model: Constant model applied to the right of the split.
        """
        super().__init__(indices, y_res, rss, pivot_idx, pivot_value,
                         left_child, right_child, left_model, right_model)

    def get_label(self) -> str:
        """Returns the full display label for this node.

        Returns:
            Label of the form "PCONC- X<pivot_idx>\\nidx ∉ <pivot_value>".
        """
        return f"PCONC- X{self.pivot_idx}\nidx ∉ {self.pivot_value}"

@BaseNode.register
class SplitCNode(SplitNode):
    """Categorical variant of SplitNode, splitting on category membership."""

    def get_label(self) -> str:
        """Returns the full display label for this node.

        Returns:
            Label of the form "SPLITC X<pivot_idx>\\nidx ∉ <pivot_value>".
        """
        return f"SPLITC X{self.pivot_idx}\nidx ∉ {self.pivot_value}"