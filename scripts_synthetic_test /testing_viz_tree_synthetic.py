print("importing PILOT...")
from pilot.pilot import PILOT
from viz_tree.viz_tree import VizTree
from viz_tree.dot_settings import DotSettings
from viz_tree.make_synthetic_datasets import synthetic_dataset

print("import done")

import numpy as np
import os
from datetime import datetime
from graphviz import Source
from subprocess import run
from sklearn.metrics import r2_score

#%%
dataset_name = "hard"
n = 500
X, y = synthetic_dataset(name=dataset_name, n=n)

pilot_model = PILOT(min_sample_leaf=50)
pilot_model.fit(X, y)
pilot_tree = pilot_model.model_tree

y_pred = pilot_model.predict(X)
print(f"R2 score = {r2_score(y, y_pred)}")

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
                       output_directory=map_directory,
                       tree_id = id_results)

#%%
count += 1
file_directory_gv =  os.path.join(map_directory_gv, f"pilot_graph_{dataset_name}_{id_results}_{count}.gv")
file_directory_pdf =  os.path.join(map_directory, f"pilot_graph_{dataset_name}_{id_results}_{count}.pdf")

dot_settings = DotSettings(rankdir = "TB",
                           combine_lin=False,
                           use_predsplot=True,
                           use_regplot=True,
                           print_model = True,
                           n_max = 4,
                           highlight_x=None,
                           staircase=True,
                           use_intercept=False)
pilot_dot = my_viz_model.get_dot(dot_settings)

src = Source(pilot_dot)
src.save(file_directory_gv)
pdf_bytes = src.pipe(format="pdf")
with open(file_directory_pdf, "wb") as f:
    f.write(pdf_bytes)
run(["open", "-a", "Preview", file_directory_pdf])