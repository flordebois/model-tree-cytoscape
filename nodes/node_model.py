from abc import ABC, abstractmethod
from typing import List
import numpy as np

class NodeModel(ABC):

    @abstractmethod
    def get_label(self) -> str:
        pass

    def to_dict(self) -> dict:
        return {"class_type": self.__class__.__name__}

    @classmethod
    @abstractmethod
    def from_dict(cls, dict):
        pass

    @abstractmethod
    def predict(self, x):
        pass

def node_model_from_dict(dict):
    class_type = globals().get(dict["class_type"])
    return class_type.from_dict(dict)

class NoneNodeModel(NodeModel):
    def get_label(self) -> str:
        return ""

    @classmethod
    def from_dict(cls, dict):
        return cls.__new__(cls)

    def predict(self, x):
        return x

class ConstantNodeModel(NodeModel):
    value: float
    def __init__(self, value):
        self.value = value

    def get_label(self) -> str:
        return f"{self.value:.3g}"

    def to_dict(self) -> dict:
        dict = super().to_dict()
        dict["value"] = self.value
        return dict

    @classmethod
    def from_dict(cls, dict):
        obj = cls.__new__(cls)
        obj.value = dict["value"]
        return obj

    def predict(self, x):
        return [self.value] * len(x)

class SimpleLinearNodeModel(NodeModel):
    idx: int
    coefficient: float
    intercept: float

    def __init__(self, idx, coefficient, intercept):
        self.idx = idx
        self.coefficient = coefficient
        self.intercept = intercept

    def get_label(self) -> str:
        return f"{self.coefficient:.3g}·X{self.idx} {self.intercept:.3g}"

    def to_dict(self) -> dict:
        dict = super().to_dict()
        dict.update({"idx": self.idx,
                     "coefficient": self.coefficient,
                     "intercept": self.intercept})
        return dict

    def predict(self, x):
        y = [self.intercept] * len(x)
        y += self.coefficient * np.array(x)
        return y

    @classmethod
    def from_dict(cls, dict):
        obj = cls.__new__(cls)
        obj.idx = dict["idx"]
        obj.coefficient = dict["coefficient"]
        obj.intercept = dict["intercept"]
        return obj

class LinearNodeModel(NodeModel):
    coefficients: List[float]
    intercept: float

    def __init__(self, coefficients, intercept):
        self.coefficients = coefficients
        self.intercept = intercept

    def get_label(self) -> str:
        parts = []
        for i, coef in enumerate(self.coefficients):
            if coef != 0:
                parts.append(f"{coef:.3g}·X{i}")
        parts.append(f"{self.intercept:.3g}")
        return " + ".join(parts)

    def to_dict(self) -> dict:
        dict = super().to_dict()
        dict.update({"coefficients": self.coefficients,
                     "intercept": self.intercept})
        return dict

    def predict(self, x):
        x_array = np.array(x)
        y = [self.intercept] * x_array.shape[0]
        y += self.coefficients @ x_array
        return y

    @classmethod
    def from_dict(cls, dict):
        obj = cls.__new__(cls)
        obj.coefficients = dict["coefficients"]
        obj.intercept = dict["intercept"]
        return obj
