from nodes.base_node import BaseNode
from typing import List

@BaseNode.register
class NoneNode(BaseNode):
    """A null object node representing an empty tree branch."""
    def __init__(self):
        # Initialize with empty/zero values
        super().__init__(indices=[], y_res=None, rss=0.0)

    def get_children(self) -> List[BaseNode]:
        return []

    def get_label(self) -> str:
        return "None"

    def get_minimal_label(self) -> str:
        return ""

    def to_dict(self) -> dict:
        return {"node_type": "NoneNode"}

    @classmethod
    def from_dict(cls, d: dict) -> "NoneNode":
        return cls()
