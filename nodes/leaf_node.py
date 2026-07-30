from typing import List
from nodes.base_node import BaseNode
from nodes.node_model import node_model_from_dict, LinearNodeModel


@BaseNode.register
class LeafNode(BaseNode):
    node_model: LinearNodeModel = None

    def __init__(self, indices, y_res, rss, node_model):
        super().__init__(indices, y_res, rss)
        self.node_model = node_model

    def get_children(self) -> List[BaseNode]:
        return []

    def get_label(self) -> str:
        return "Leaf\n" + self.node_model.get_label()

    def get_minimal_label(self) -> str:
        return "L"

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["node_model"] = self.node_model.to_dict()
        return node_dict

    @classmethod
    def from_dict(cls, dict):
        obj = super().from_dict(dict)
        obj.node_model = node_model_from_dict(dict["node_model"])
        return obj