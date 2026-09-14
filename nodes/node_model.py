"""Prediction models attached to tree nodes.

A NodeModel encapsulates how a node turns input features into a
prediction (e.g. a constant, a one dimensional linear model, or a
full linear model).
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np

class NodeModel(ABC):
    """Abstract base class for all node prediction models."""

    @abstractmethod
    def get_label(self) -> str:
        """Returns a short display label describing the model.

        Returns:
            String of label
        """
        pass

    def to_dict(self) -> dict:
        """Serializes the model to a dictionary.

        Subclasses should call this and extend the result with their
        own parameters.

        Returns:
            Dictionary containing at least the model's class name
            under "class_type".
        """
        return {"class_type": self.__class__.__name__}

    @classmethod
    @abstractmethod
    def from_dict(cls, dic: dict) -> "NodeModel":
        """Reconstructs a model instance from a dictionary.

        Args:
            dic: Dictionary of NodeModel previously produced by to_dict().

        Returns:
            A new instance of cls with parameters restored.
        """
        pass

    @abstractmethod
    def predict(self, x):
        """Predicts the target value(s) for the given input.

        Args:
            x: Input feature(s) to predict on.

        Returns:
            Predicted value(s).
        """
        pass

def node_model_from_dict(dic: dict) -> NodeModel:
    """Reconstructs a NodeModel instance of the correct subclass.

    Args:
        dic: Dictionary previously produced by a NodeModel's to_dict(),
            containing a "class_type" key.

    Returns:
        A new NodeModel instance of the matching subclass.
    """
    class_type = globals().get(dic["class_type"])
    return class_type.from_dict(dic)

class NoneNodeModel(NodeModel):
    """Identity model used as a placeholder when no model is applied."""

    def get_label(self) -> str:
        """Returns an empty label, since no model is applied.

        Returns:
            Empty string.
        """
        return ""

    @classmethod
    def from_dict(cls, dic: dict) -> "NoneNodeModel":
        """Reconstructs a NoneNodeModel instance.

        Args:
            dic: Unused; present for interface consistency.

        Returns:
            A new NoneNodeModel instance.
        """
        return cls.__new__(cls)

    def predict(self, x):
        """Returns the input unchanged.

        Args:
            x: Input feature(s).

        Returns:
            x, unmodified.
        """
        return x

class ConstantNodeModel(NodeModel):
    """Model that predicts a single constant value.

    Attributes:
        value: The constant value predicted for any input.
    """

    value: float
    def __init__(self, value: float):
        """
        Args:
            value: The constant value to predict.
        """
        self.value = value

    def get_label(self) -> str:
        """Returns the constant value as a display label.

        Returns:
            The value formatted to 3 significant digits.
        """
        return f"{self.value:.3g}"

    def to_dict(self) -> dict:
        """Serializes the model to a dictionary.

        Returns:
            Dictionary with class_type and value.
        """
        dic = super().to_dict()
        dic["value"] = self.value
        return dic

    @classmethod
    def from_dict(cls, dic: dict) -> "ConstantNodeModel":
        """Reconstructs a ConstantNodeModel instance.

        Args:
            dic: Dictionary previously produced by to_dict().

        Returns:
            A new ConstantNodeModel instance.
        """
        obj = cls.__new__(cls)
        obj.value = dic["value"]
        return obj

    def predict(self, x: List[float]) -> List[float]:
        """Predicts the constant value for each input row.

        Args:
            x: Input feature(s); only its length is used.

        Returns:
            List containing the constant value, repeated len(x) times.
        """
        return [self.value] * len(x)

class SimpleLinearNodeModel(NodeModel):
    """Model with a one dimensional linear regression model.

    Attributes:
        idx: Index of the feature used in the linear term.
        coefficient: Coefficient applied to the feature at idx.
        intercept: Constant offset added to the prediction.
    """

    idx: int
    coefficient: float
    intercept: float

    def __init__(self, idx: int, coefficient: float, intercept: float):
        """
        Args:
            idx: Index of the feature used in the linear term.
            coefficient: Coefficient applied to the feature at idx.
            intercept: Constant offset added to the prediction.
        """
        self.idx = idx
        self.coefficient = coefficient
        self.intercept = intercept

    def get_label(self) -> str:
        """Returns the linear term as a display label.

        Returns:
            Label of the linear model.
        """
        return f"{self.coefficient:.3g}·X{self.idx} + {self.intercept:.3g}"

    def to_dict(self) -> dict:
        """Serializes the model to a dictionary.

        Returns:
            Dictionary with class_type, idx, coefficient, and intercept.
        """
        dic = super().to_dict()
        dic.update({"idx": self.idx,
                     "coefficient": self.coefficient,
                     "intercept": self.intercept})
        return dic

    def predict(self, x: List[float]) -> List[float]:
        """Predicts using the one dimensional linear regression model.

        Args:
            x: List of feature values.

        Returns:
            List of predicted values.
        """
        y = [self.intercept] * len(x)
        y += self.coefficient * np.array(x)
        return y

    @classmethod
    def from_dict(cls, dic: dict) -> "SimpleLinearNodeModel":
        """Reconstructs a SimpleLinearNodeModel instance.

        Args:
            dic: Dictionary previously produced by to_dict().

        Returns:
            A new SimpleLinearNodeModel instance.
        """
        obj = cls.__new__(cls)
        obj.idx = dic["idx"]
        obj.coefficient = dic["coefficient"]
        obj.intercept = dic["intercept"]
        return obj

class LinearNodeModel(NodeModel):
    """Model with a full linear regression model.

    Attributes:
        coefficients: Coefficient for each feature.
        intercept: Constant offset added to the prediction.
    """

    coefficients: List[float]
    intercept: float

    def __init__(self, coefficients: List[float], intercept: float):
        """
        Args:
            coefficients: Coefficient for each feature.
            intercept: Constant offset added to the prediction.
        """
        self.coefficients = coefficients
        self.intercept = intercept

    def get_label(self) -> str:
        """Returns the linear model as a display label.

        Only features with a non-zero coefficient are included.

        Returns:
            String of the linear model.
        """
        parts = []
        for i, coef in enumerate(self.coefficients):
            if coef != 0:
                parts.append(f"{coef:.3g}·X{i}")
        parts.append(f"{self.intercept:.3g}")
        return " + ".join(parts)

    def to_dict(self) -> dict:
        """Serializes the model to a dictionary.

        Returns:
            Dictionary with class_type, coefficients, and intercept.
        """
        dic = super().to_dict()
        dic.update({"coefficients": self.coefficients,
                     "intercept": self.intercept})
        return dic

    def predict(self, x: List[float]) -> List[float]:
        """Predicts using the full linear regression model.

        Args:
            x: Input feature matrix, one row per sample.

        Returns:
            Array of predicted values.
        """
        x_array = np.array(x)
        y = [self.intercept] * x_array.shape[0]
        y += self.coefficients @ x_array
        return y

    def add_model(self, model: NodeModel):
        """Merges another model's parameters into this one, in place.

        Args:
            model: A SimpleLinearNodeModel, ConstantNodeModel, or
                LinearNodeModel to merge in.

        Raises:
            ValueError: If model is a LinearNodeModel with a different
                number of coefficients than this model.
        """
        if isinstance(model, SimpleLinearNodeModel):
            self.coefficients[model.idx] = model.coefficient
            self.intercept += model.intercept
        elif isinstance(model, ConstantNodeModel):
            self.intercept += model.value
        elif isinstance(model, LinearNodeModel):
            if len(self.coefficients) != len(model.coefficients):
                raise ValueError(f"Number of coefficients does not match: {len(self.coefficients)} vs {len(model.coefficients)}")
            self.coefficients = model.coefficients
            self.intercept += model.intercept

    @classmethod
    def from_dict(cls, dic: dict) -> "LinearNodeModel":
        """Reconstructs a LinearNodeModel instance.

        Args:
            dic: Dictionary previously produced by to_dict().

        Returns:
            A new LinearNodeModel instance.
        """
        obj = cls.__new__(cls)
        obj.coefficients = dic["coefficients"]
        obj.intercept = dic["intercept"]
        return obj