import sys
import os
import xgboost as xgb
from xgboost import plot_importance, plot_tree, plotting

import dtreeviz
import graphviz
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

import pandas as pd
import numpy as np

random_state = 1234 # get reproducible trees

#%%
dataset_url = "https://raw.githubusercontent.com/parrt/dtreeviz/master/data/titanic/titanic.csv"
dataset = pd.read_csv(dataset_url)
# Fill missing values for Age
dataset.fillna({"Age":dataset.Age.mean()}, inplace=True)
# Encode categorical variables
dataset["Sex_label"] = dataset.Sex.astype("category").cat.codes
dataset["Cabin_label"] = dataset.Cabin.astype("category").cat.codes
dataset["Embarked_label"] = dataset.Embarked.astype("category").cat.codes

#%%
features = ["Pclass", "Age", "Fare", "Sex_label", "Cabin_label", "Embarked_label"]
target = "Survived"

dtrain = xgb.DMatrix(dataset[features], dataset[target])

params = {"max_depth":3, "eta":0.05, "objective":"binary:logistic", "subsample":1}
xgb_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=8)

#%%
viz_model = dtreeviz.model(xgb_model, tree_index=1,
                           X_train=dataset[features], y_train=dataset[target],
                           feature_names=features,
                           target_name=target, class_names=["perish", "survive"])

#%%
from types import MethodType
from graphviz import Source
import tempfile, webbrowser

def my_view(self, *args, **kwargs):
    """Custom dtreeviz viewer: renders SVG and opens it in browser."""
    render = self.view(*args, **kwargs)
    dot = render.dot
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".svg")
    Source(dot).render(tmp.name, format="svg", cleanup=True)
    webbrowser.open(f"file://{tmp.name}.svg")
    return tmp.name + ".svg"  # return the file path in case you want it

# attach the method to your existing viz_model
viz_model.my_view = MethodType(my_view, viz_model)

#%%
viz_model.my_view(depth_range_to_display=(1, 2))


#%%
viz_model.my_view(orientation="LR")

#%%
viz_model.my_view(fancy=False)

#%%
viz_model.my_view(depth_range_to_display=(1, 2)) # root is level 0

#%%
x = dataset[features].iloc[10]
x

#%%
viz_model.my_view(x=x)
#%%
viz_model.my_view(x=x, show_just_path=True)

#%%
print(viz_model.explain_prediction_path(x))

#%%
viz_model.leaf_sizes()
#%%
viz_model.ctree_leaf_distributions()
#%%
viz_model.node_stats(node_id=10)

#%%
features_reg = ["Pclass", "Fare", "Sex_label", "Cabin_label", "Embarked_label", "Survived"]
target_reg = "Age"

dtrain_reg = xgb.DMatrix(dataset[features_reg], dataset[target_reg])
params_reg = {"max_depth":3, "eta":0.05, "objective":"reg:squarederror", "subsample":1}
xgb_model_reg = xgb.train(params=params_reg, dtrain=dtrain_reg, num_boost_round=8)
#%% md
# ## Initialize dtreeviz model (adaptor)
#%%
viz_rmodel = dtreeviz.model(model=xgb_model_reg, tree_index=1,
                            X_train=dataset[features_reg],
                            y_train=dataset[target_reg],
                            feature_names=features_reg,
                            target_name=target_reg)

viz_rmodel.my_view = MethodType(my_view, viz_rmodel)
#%% md
# ## Tree structure visualisations
#%%
viz_rmodel.my_view()
#%%
viz_rmodel.my_view(orientation="LR")
#%%
viz_rmodel.my_view(fancy=False)
#%%
viz_rmodel.my_view(depth_range_to_display=(0, 2))
#%% md
# ## Prediction path explanations
#
#%%
x = dataset[features_reg].iloc[10]
x
#%%
viz_rmodel.my_view(x = x)
#%%
viz_rmodel.my_view(show_just_path=True, x = x)
#%%
print(viz_rmodel.explain_prediction_path(x))
#%% md
# ## Leaf info
#%%
viz_rmodel.leaf_sizes()
#%%
viz_rmodel.rtree_leaf_distributions()
#%%
viz_rmodel.node_stats(node_id=4)