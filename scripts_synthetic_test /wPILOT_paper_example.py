print("Importing PILOT...")
from pilot.pilot import PILOT
from viz_tree.viz_tree import VizTree
from viz_tree.dot_settings import DotSettings

print("Import done")
import numpy as np
import pandas as pd
import os
from datetime import datetime
from graphviz import Source
from subprocess import run

#%%
np.random.seed(43)
n = 1500
pd_X = pd.DataFrame({
    'X0': np.concatenate([np.random.uniform(0, 5, int(1/2*n)),
                    np.random.uniform(5.2, 10, int(1/2*n))]),
    'X1': np.random.uniform(14, 26, n),
    'X2': np.random.uniform(25, 75, n),
    'X3': np.random.uniform(0, 10, n)
})

def get_response(r):
    if r['X1'] <= 18:
        # Left-most Leaf
        return 3*r['X1'] + 0.25*r['X2'] - 5*r['X3'] + 5
    elif r['X0'] > 5:
        # Far-right Leaf
        return -3*r['X1'] + 10*r['X0']
    elif r['X2'] > 50:
        # Middle-right Leaf
        return -3*r['X1'] - 2*r['X0'] + 50
    else:
        # Bottom-left Leaf
        return -3*r['X1'] - 2*r['X0'] + 2*r['X3'] + 62

pd_y = pd_X.apply(get_response, axis=1) + np.random.normal(0, 2, n)

n2 = 3000
pd_X2 = pd.DataFrame({
        'X0': np.concatenate([np.random.uniform(0, 5, int(1/2*n2)),
                    np.random.uniform(5.2, 10, int(1/2*n2))]),
    'X1': np.random.uniform(14, 26, n2),
    'X2': np.random.uniform(25, 75, n2),
    'X3': np.random.uniform(0, 10, n2)
})

def get_response2(r):
    if r['X1'] <= 18:
        # Left-most Leaf
        return -2.8*r['X1'] + 0.4*r['X2'] - 5*r['X3'] + 37
    elif r['X0'] > 5:
        # Far-right Leaf
        return -3*r['X1'] + 10*r['X0']
    else:
        # Bottom-left Leaf
        return -3*r['X1'] - 2*r['X0'] + 2*r['X3'] + 58

pd_y2 = pd_X2.apply(get_response2, axis=1) + np.random.normal(0, 4, n2)

#%%
X1 = pd_X.to_numpy()
y1 = pd_y.to_numpy()
X2 = pd_X2.to_numpy()
y2 = pd_y2.to_numpy()
X = np.concatenate([X1, X2])
y = np.concatenate([y1, y2])

w = np.ones_like(y)
w[:1500] = 20 #Turn on or off depending on if you wnat to use weigths

model = PILOT(max_model_depth=4, max_depth=3)
model.fit(X, y, w)
pilot_tree = model.model_tree

#%%
dataset_name = "synthetic_dataset"
output_directory = "/Users/flor/Pycharm/PILOT/output/synthetic_example"
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
                           use_predsplot=False,
                           use_regplot=True,
                           print_model = True,
                           n_max = 4,
                           highlight_x=None,
                           staircase=False,
                           use_intercept=False)
pilot_dot = my_viz_model.get_dot(dot_settings)

src = Source(pilot_dot)
src.save(file_directory_gv)
pdf_bytes = src.pipe(format="pdf")
with open(file_directory_pdf, "wb") as f:
    f.write(pdf_bytes)
run(["open", "-a", "Preview", file_directory_pdf])

