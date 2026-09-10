from abc import ABC, abstractmethod
import numpy as np
from nodes.base_node import BaseNode

ADAPTERS_REGISTRY: dict[str, "BaseAdapter"] = {}

def register_adapter(name):
    def decorator(adapter_cls):
        ADAPTERS_REGISTRY[name] = adapter_cls
        return adapter_cls
    return decorator

class BaseAdapter(ABC):
    @staticmethod
    @abstractmethod
    def build_root_node(X_train: np.ndarray, y_train: np.ndarray, model) -> BaseNode:
        pass

    @staticmethod
    @abstractmethod
    def load_model(model_path):
        pass

    @staticmethod
    def predict(X: np.ndarray, model) -> np.ndarray | None:
        # Optional. Return None if this adapter can't cheaply predict,
        # VizTree will computing y_hat from the built tree.
        return None




