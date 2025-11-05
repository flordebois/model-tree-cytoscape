print("importing PILOT...")
from pilot.pilot import PILOT
from pilot.copilot import coPILOT
from pilot.viz_tree import VizTree

print("import done")

import numpy as np
import matplotlib.pyplot as plt
import dtreeviz
import xgboost as xgb
import os
from datetime import datetime
from pmlb import fetch_data
from graphviz import Source
from subprocess import run

#%%
dataset_name = "294_satellite_image" # "547_no2" "294_satellite_image" #"658_fri_c3_250_25"
data = fetch_data(dataset_name)
X = np.array(data.iloc[:, :-1])
y = np.array(data.iloc[:, -1])
feature_names = data.iloc[:, :-1].columns.tolist()
target_name = "Target"

#%%
pilot_model = PILOT(min_sample_leaf=25, max_model_depth=5)
pilot_model.fit(X, y)
pilot_tree = pilot_model.model_tree

# copilot_model = coPILOT(max_model_depth=4, max_n_estimators=2, alpha=0.5)
# copilot_model.fit(X, y, stop_early=False)
# pilot_tree = copilot_model.pilot_trees[0].model_tree

#%%
output_directory = "/Users/flor/Pycharm/PILOT-VIS/viz_tree/output"
map_directory = os.path.join(output_directory, dataset_name)
os.makedirs(map_directory, exist_ok=True)
map_directory_gv = os.path.join(map_directory, "gv_files")
os.makedirs(map_directory_gv, exist_ok=True)

my_viz_model = VizTree(tree_model = pilot_tree,
                       X_train=X,
                       y_train=y,
                       rankdir = "LR",
                       feature_names=feature_names,
                       target_name=target_name,
                       is_predsplot_leafs=True,
                       is_regplot_nodes=True,
                       output_directory=map_directory)

#%%
id_results = datetime.now().strftime('%d-%m-%y_%H-%M-%S')

file_directory_gv =  os.path.join(map_directory_gv, f"pilot_graph_{dataset_name}_{id_results}.gv")
file_directory_pdf =  os.path.join(map_directory, f"pilot_graph_{dataset_name}_{id_results}.pdf")

pilot_dot = my_viz_model.get_dot(combine_lin=False)
src = Source(pilot_dot)
src.save(file_directory_gv)
pdf_bytes = src.pipe(format="pdf")
with open(file_directory_pdf, "wb") as f:
    f.write(pdf_bytes)
run(["open", "-a", "Preview", file_directory_pdf])

# #%%
# dtrain_reg = xgb.DMatrix(X, y)
# params_reg = {"max_depth":2, "eta":0.05, "objective":"reg:squarederror", "subsample":1, "min_child_weight":1}
# xgb_model_reg = xgb.train(params=params_reg, dtrain=dtrain_reg, num_boost_round=8)
# #%%
# viz_rmodel = dtreeviz.model(model=xgb_model_reg, tree_index=1,
#                             X_train=X,
#                             y_train=y,
#                             feature_names=feature_names,
#                             target_name=target_name)
#
# #%%
# dot = viz_rmodel.view().dot
#
# file_directory_gv2 =  os.path.join(map_directory_gv, f"pilot_dtreeviz_{dataset_name}_{id_results}.gv")
# file_directory_pdf2 =  os.path.join(map_directory, f"pilot_dtreeviz_{dataset_name}_{id_results}.pdf")
#
# src = Source(dot)
# src.save(file_directory_gv2)
# pdf_bytes = src.pipe(format="pdf")
# with open(file_directory_pdf2, "wb") as f:
#     f.write(pdf_bytes)
# run(["open", "-a", "Preview", file_directory_pdf2])
