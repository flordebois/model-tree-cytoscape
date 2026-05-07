from abc import ABC, abstractmethod
import numpy as np
from typing import List


# NODE_SHAPES = {"leaf": "ellipse", "lin": "ellipse", "blin": "box", "pcon": "box", "plin": "box", "pconc": "box"}
# NODE_LABEL = {"leaf": "Leaf", "lin": "LIN", "blin": "BLIN", "pcon": "PCON", "plin": "PLIN", "pconc": "PCONC"}
# NODE_FILL_COLOR = {"leaf": "#2ca02c", "lin": "#9467bd", "blin": "#1f77b4", "pcon": "#d62728", "plin": "#ff7f0e", "pconc": "#8c564b"}
# NODE_STYLE = {"leaf": "filled", "lin": "filled", "blin": "rounded,filled", "pcon": "rounded,filled", "plin": "filled", "pconc": "rounded,filled"}
# NODE_FONT_NAME = "Arial"
# NODE_FONT_COLOR = "white"

class BaseNode(ABC):
    id: int = None
    type: str = None
    indices: np.ndarray = None
    y_res: np.ndarray = None
    
    def set_id(self, id: int):
        self.id = id

    @abstractmethod
    def get_children(self):
        pass

    @abstractmethod
    def get_label(self) -> str:
        pass

    @abstractmethod
    def get_minimal_label(self) -> str:
        pass


    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "indices": self.indices.tolist(),
            "y_res": self.y_res.tolist(),
        }

class LeafNode(BaseNode):
    def __init__(self, indices, y_res, coefficients, intercept):
        self.type = "leaf"
        self.indices = indices
        self.y_res = y_res
        self.coefficients = coefficients
        self.intercept = intercept

    def get_children(self) -> List[BaseNode]:
        return []

    def get_label(self) -> str:
        parts = []
        for i, coef in enumerate(self.coefficients):
            if coef != 0:
                parts.append(f"{coef:.3g}·X{i}")
        parts.append(f"{self.intercept:.3g}")
        return "Leaf\n" + " + ".join(parts)

    def get_minimal_label(self) -> str:
        return ""

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict.update({
                "coefficients": self.coefficients.tolist(),
                "intercept": self.intercept,
            })
        return node_dict

class InternalNode(BaseNode):
    def __init__(self, type, indices, y_res, pivot_idx, pivot_value, left_lin_model, left_child_node,
                 right_lin_model, right_child_node):
        self.type = type
        self.indices = indices
        self.y_res = y_res
        self.pivot_idx = pivot_idx
        self.pivot_value = pivot_value
        self.left_child_node = left_child_node
        self.left_lin_model = left_lin_model
        self.right_child_node = right_child_node
        self.right_lin_model = right_lin_model

    def get_children(self) -> List[BaseNode]:
        children = [self.left_child_node]
        if self.right_child_node is not None:
            children.append(self.right_child_node)
        return children

    def get_label(self) -> str:
        ntype = self.type.upper()
        if self.type == "lin":
            return f"{ntype}\nX{self.pivot_idx}"
        return f"{ntype}\nX{self.pivot_idx} > {self.pivot_value:.3g}"

    def get_minimal_label(self) -> str:
        return f"X{self.pivot_idx}"

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict.update({
                "pivot_idx": self.pivot_idx,
                "pivot_value": self.pivot_value,
                "left_lin_model": list(self.left_lin_model) if self.left_lin_model is not None else None,
                "right_lin_model": list(self.right_lin_model) if self.right_lin_model is not None else None,
                "left_child": self.left_child_node.id,
                "right_child": self.right_child_node.id if self.right_child_node else None,
            })
        return node_dict

class CombinedLinNode(BaseNode):
    def __init__(self, nodes: List[InternalNode]):
        pivot_indices = []
        lin_coefficients = {}
        intercept = 0
        prev_node = None
        for node in nodes:
            # check that nodes are a chain of linear nodes
            if node.type != "lin":
                raise ValueError("One of the nodes is not linear")
            if prev_node is not None:
                if prev_node.left_child_node != node:
                    raise ValueError("Nodes are not a chain of linear nodes")
            prev_node = node

            pivot_indices.append(node.pivot_idx)
            lin_coefficients[node.pivot_idx] = node.left_lin_model[0]
            intercept += node.left_lin_model[1]

        self.child = nodes[-1].left_child_node
        self.pivot_indices = pivot_indices
        self.lin_coefficients = lin_coefficients
        self.intercept = intercept
        self.type = "combined_lin"
        self.indices = nodes[-1].indices
        self.y_res = nodes[-1].y_res

    def get_children(self):
        return [self.child]

    def get_label(self) -> str:
        idx_str = ", ".join(str(i) for i in self.pivot_indices)
        return f"LIN\nidx={idx_str}"

    def get_minimal_label(self) -> str:
        idx_str = ",".join(str(i) for i in self.pivot_indices[:1])
        return f"X{idx_str}.."

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict.update({
                "pivot_indices": self.pivot_indices,
                "lin_coefficients": self.lin_coefficients,
                "intercept": self.intercept,
                "child": self.child.id,
            })
        return node_dict