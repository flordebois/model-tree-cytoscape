from abc import ABC, abstractmethod
import numpy as np
from typing import List

class BaseNode(ABC):
    id: int = None
    indices: np.ndarray = None
    y_res: np.ndarray = None
    rss: float = None

    def __init__(self, indices, y_res, rss):
        self.indices = indices
        self.y_res = y_res
        self.rss = rss

    def set_id(self, id: int):
        self.id = id

    @abstractmethod
    def get_children(self):
        pass

    def get_all_children(self):
        return self.get_children() + sum([child.get_all_children() for child in self.get_children()], [])

    @abstractmethod
    def get_label(self) -> str:
        pass

    @abstractmethod
    def get_minimal_label(self) -> str:
        pass

    def to_dict(self):
        node_dict = {"id": self.id, "rss": self.rss}
        if isinstance(self.indices, List):
            node_dict["indices"] = self.indices
        else:
            node_dict["indices"] = self.indices.tolist()
        if isinstance(self.y_res, List):
            node_dict["y_res"] = self.y_res
        else:
            node_dict["y_res"] = self.y_res.tolist()
        return node_dict

    @classmethod
    def from_dict(cls, node_dict):
        obj = cls.__new__(cls)
        obj.id = node_dict["id"]
        obj.indices = node_dict["indices"]
        obj.y_res = node_dict["y_res"]
        obj.rss = node_dict["rss"]
        return obj

    class_registry = {}
    @classmethod
    def register(cls, subclass):
        cls.class_registry[subclass.__name__] = subclass
        return subclass