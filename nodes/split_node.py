from typing import List

from nodes.base_node import BaseNode
from nodes.node_model import NodeModel, ConstantNodeModel, node_model_from_dict
from nodes.internal_node import InternalNode


@BaseNode.register
class SplitNode(InternalNode):
    """Base split node. Defaults to NoneNodeModel if no specific model is applied."""
    pivot_value: float
    left_child: BaseNode
    right_child_node: BaseNode
    left_model: NodeModel
    right_model: NodeModel

    def __init__(self, indices, y_res, rss, pivot_idx: int, pivot_value,
                 left_child: BaseNode, right_child: BaseNode,
                 left_model: NodeModel, right_model: NodeModel):
        super().__init__(indices, y_res, rss, pivot_idx)
        self.pivot_value = pivot_value
        self.left_child = left_child
        self.right_child = right_child
        self.left_model = left_model
        self.right_model = right_model

    def get_children(self) -> List[BaseNode]:
        return [self.left_child, self.right_child]

    def set_children(self, children: List[BaseNode]):
        self.left_child = children[0]
        self.right_child = children[1]

    def get_label(self) -> str:
        # Default label for pure split nodes
        return f"SPLIT\nX{self.pivot_idx} > {self.pivot_value:.3g}"

    def to_dict(self) -> dict:
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
    def from_dict(cls, dict) -> "SplitNode":
        left_child_class = cls.class_registry[dict["left_child_class"]]
        right_child_class = cls.class_registry[dict["right_child_class"]]

        obj = super().from_dict(dict)
        obj.pivot_idx = dict["pivot_idx"]
        obj.pivot_value = dict["pivot_value"]
        obj.left_child = left_child_class.from_dict(dict["left_child_dict"])
        obj.right_child = right_child_class.from_dict(dict["right_child_dict"])
        obj.left_model = node_model_from_dict(dict["left_model"])
        obj.right_model = node_model_from_dict(dict["right_model"])
        return obj

@BaseNode.register
class PlinNode(SplitNode):
    def get_label(self) -> str:
        return f"PLIN\nX{self.pivot_idx} > {self.pivot_value:.3g}"

@BaseNode.register
class BlinNode(SplitNode):
    def get_label(self) -> str:
        return f"BLIN\nX{self.pivot_idx} > {self.pivot_value:.3g}"

@BaseNode.register
class PconNode(SplitNode):
    def get_label(self) -> str:
        return f"PCON\nX{self.pivot_idx} > {self.pivot_value:.3g}"

@BaseNode.register
class PconcNode(PconNode):
    def __init__(self, indices, y_res, rss, pivot_idx: int, pivot_value,
                 left_child: BaseNode, right_child: BaseNode,
                 left_model: ConstantNodeModel, right_model: ConstantNodeModel):
        super().__init__(indices, y_res, rss, pivot_idx, pivot_value,
                         left_child, right_child, left_model, right_model)

    def get_label(self) -> str:
        return f"PCONC- X{self.pivot_idx}\nleft:{self.pivot_value}"