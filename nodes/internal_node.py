from abc import ABC, abstractmethod
from typing import List

from nodes.base_node import BaseNode
from nodes.node_model import SimpleLinearNodeModel

class InternalNode(BaseNode, ABC):
    """Abstract base for all internal/split nodes."""
    pivot_idx: int

    def __init__(self, indices, y_res, rss, pivot_idx: int):
        super().__init__(indices, y_res, rss)
        self.pivot_idx = pivot_idx

    def get_minimal_label(self) -> str:
        return f"X{self.pivot_idx}"

    @abstractmethod
    def set_children(self, children: List[BaseNode]):
        pass

@BaseNode.register
class LinearNode(InternalNode):
    """LIN: Single-child node containing one linear model."""
    linear_model: SimpleLinearNodeModel
    child: BaseNode

    def __init__(self, indices, y_res, rss, pivot_idx: int,
                 linear_model: SimpleLinearNodeModel, child: BaseNode):
        super().__init__(indices, y_res, rss, pivot_idx)
        self.linear_model = linear_model
        self.child = child

    def get_children(self) -> List[BaseNode]:
        return [self.child]

    def set_children(self, children: List[BaseNode]):
        self.child = children[0]

    def get_label(self) -> str:
        return f"LIN\nX{self.pivot_idx}"

    def to_dict(self) -> dict:
        node_dict = super().to_dict()
        node_dict.update({
            "pivot_idx": self.pivot_idx,
            "linear_model": self.linear_model.to_dict(),
            "child_class": self.child.__class__.__name__,
            "child_dict": self.child.to_dict()
        })
        return node_dict

    @classmethod
    def from_dict(cls, dict) -> "LinearNode":
        child_class = cls.class_registry[dict["child_class"]]

        obj = super().from_dict(dict)
        obj.pivot_idx = dict["pivot_idx"]
        obj.linear_model = SimpleLinearNodeModel.from_dict(dict["linear_model"])
        obj.child = child_class.from_dict(dict["child_dict"])
        return obj
