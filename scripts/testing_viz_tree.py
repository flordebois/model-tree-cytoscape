print("importing PILOT...")
from pilot.pilot import PILOT
from pilot.copilot import coPILOT
from viz_tree.viz_tree import VizTree

print("import done")

import numpy as np
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
pilot_model = PILOT() #min_sample_leaf=25, max_model_depth=3)
pilot_model.fit(X, y)
pilot_tree = pilot_model.model_tree

# copilot_model = coPILOT(max_model_depth=4, max_n_estimators=2, alpha=0.5)
# copilot_model.fit(X, y, stop_early=False)
# pilot_tree = copilot_model.pilot_trees[0].model_tree

#%%
output_directory = "/Users/flor/Pycharm/PILOT-VIS/scripts/output"
map_directory = os.path.join(output_directory, dataset_name)
os.makedirs(map_directory, exist_ok=True)
map_directory_gv = os.path.join(map_directory, "gv_files")
os.makedirs(map_directory_gv, exist_ok=True)
id_results = datetime.now().strftime('%d-%m-%y_%H-%M-%S')
count=0
my_viz_model = VizTree(pilot_tree = pilot_tree,
                       X_train=X,
                       y_train=y,
                       rankdir = "TB",
                       feature_names=feature_names,
                       target_name=target_name,
                       output_directory=map_directory,
                       tree_id = id_results)

#%%
count += 1
file_directory_gv =  os.path.join(map_directory_gv, f"pilot_graph_{dataset_name}_{id_results}_{count}.gv")
file_directory_pdf =  os.path.join(map_directory, f"pilot_graph_{dataset_name}_{id_results}_{count}.pdf")

highlight_x = X[20,:]
highlight_x[19] = 95
highlight_x[20] = 59
highlight_x[17] = 56
pilot_dot = my_viz_model.get_dot(combine_lin=True,
                                 use_predsplot=False,
                                 use_regplot=False,
                                 highlight_x=highlight_x)
src = Source(pilot_dot)
src.save(file_directory_gv)
pdf_bytes = src.pipe(format="pdf")
with open(file_directory_pdf, "wb") as f:
    f.write(pdf_bytes)
run(["open", "-a", "Preview", file_directory_pdf])
