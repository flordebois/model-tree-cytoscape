# Example: the partykit adapter

`partykit` is an R package for fitting model-based recursive partitioning trees [[1]](#1). Its `lmtree` function
implements the MOB model tree algorithm [[2]](#2). This example shows how an adapter is written to load the tree fitted
by the MOB into the application. 

## Fit and export the model in R
The package has no native or intuitive export system that is easy to use, so a simple list was created
with all the relevant info and saved as a JSON file.

```r
library(partykit)
library(jsonlite)

# Load PMLB no2 dataset
library(pmlbr)
no2 <- fetch_data("547_no2")

# Fit linear model tree to all the data
tree <- lmtree(target ~ . | ., data = no2)

# Write tree opbject to JSON file
nodes <- lapply(nodeids(tree), function(id) {
  nd <- tree[id]
  node_obj <- node_party(nd)
  is_terminal <- is.terminal(node_obj)
  
  node_info <- list(
    id <- id-1,
    is_terminal <- is_terminal,
    n <- info_node(node_obj)$nobs
  )
  
  if (is_terminal) {
    # model coefficients at this leaf
    coefs <- coef(tree, node = id)
    node_info$coefficients <- as.list(coefs)
  } else {
    split <- split_node(node_obj)
    varid <- split$varid
    varname <- names(data_party(tree))[varid]
    node_info$split_var <- varname
    node_info$breaks <- split$breaks
    node_info$index <- split$index
    node_info$levels <- levels(tree$data[[varname]])
    node_info$kids <- id-1 + c(kids_node(node_obj)[[1]]$id-1,kids_node(node_obj)[[2]]$id-1)
  }
  node_info
})

tree_json <- list(
  names <- names(tree$data),
  nodes <- nodes
  )

write_json(tree_json, "..path/tree_547_no2.json", auto_unbox = TRUE, null = "null")

```

The JSON is will look like:

```json
{
  "names": ["y", "x1", "x2", "..."],
  "nodes": [
    {
      "is_terminal": false,
      "split_var": "x1",
      "breaks": 0.5,
      "kids": [1, 2]
    },
    {
      "is_terminal": true,
      "coefficients": {"(Intercept)": 1.2, "x2": 0.3}
    }
  ]
}
```

Under `names` the feature names are stored in the same column order as `X_train`.  Each entry in `nodes` is either a
split node (`split_var` with either `breaks` for a numeric threshold or `index`/`levels` for a categorical split) or 
a leaf node (`is_terminal: true`, with a linear model's `coefficients`).

## Define the new adapter

```python
@register_adapter('Partykit')
class PartyKitAdapter(BaseAdapter):

    @staticmethod
    def load_model(model_path):
        with open(model_path) as f:
            return json.load(f)

    @staticmethod
    def build_root_node(X_train, y_train, model):
        root_indices = np.ones(X_train.shape[0], dtype=bool)
        node_list = model["nodes"]
        feature_names = model["names"][1:]
        return PartyKitAdapter._build_recursive(
            node_list, feature_names, node_list[0], X_train, y_train, root_indices
        )
```

See [`partykit_adapter.py`](../reference/adapters/partykit_adapter.md) for
the full recursive implementation.

The Partykit adapter doesn't override `predict` so `VizTree` computes its predictions by walking the built tree.

## References
<a id="1">[1]</a> 
Hothorn, T., & Zeileis, A. (2015). partykit: A modular toolkit for recursive partytioning in R. The Journal of Machine Learning Research, 16(1), 3905-3909.

<a id="2">[2]</a> 
Zeileis, A., Hothorn, T., & Hornik, K. (2008). Model-based recursive partitioning. Journal of Computational and Graphical Statistics, 17(2), 492-514.