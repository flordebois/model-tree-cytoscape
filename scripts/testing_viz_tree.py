print("importing PILOT...")
from pilot.pilot import PILOT
from pilot.copilot import coPILOT
from viz_tree.viz_tree import VizTree
from viz_tree.dot_settings import DotSettings

print("import done")

import numpy as np
import os
from datetime import datetime
from pmlb import fetch_data
from graphviz import Source
from subprocess import run

#%%
dataset_name = "547_no2" # "547_no2" "294_satellite_image" #"658_fri_c3_250_25"
data = fetch_data(dataset_name)
X = np.array(data.iloc[:, :-1])
y = np.array(data.iloc[:, -1])
feature_names = data.iloc[:, :-1].columns.tolist()
feature_names[3] = "temperature_diff"
target_name = "Target"

#%%
# pilot_model = PILOT(max_model_depth=3)
# pilot_model.fit(X, y)
# pilot_tree = pilot_model.model_tree

copilot_model = coPILOT(max_model_depth=4, max_n_estimators=2, alpha=0.5, min_sample_leaf=50)
copilot_model.fit(X, y, stop_early=False)
pilot_tree = copilot_model.pilot_trees[0].model_tree

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

point = X[383,:].copy()
point[1] = 4.5
point[2] = 2.6
point[3] = 2.5
dot_settings = DotSettings(rankdir = "TB",
                           feature_names=feature_names,
                           combine_lin=False,
                           use_predsplot=True,
                           use_regplot=True,
                           n_max = 5,
                           highlight_x=point,
                           staircase=True,
                           use_intercept=None,
                           print_model = False)
pilot_dot = my_viz_model.get_dot(dot_settings)

src = Source(pilot_dot)
src.save(file_directory_gv)
pdf_bytes = src.pipe(format="pdf")
with open(file_directory_pdf, "wb") as f:
    f.write(pdf_bytes)
run(["open", "-a", "Preview", file_directory_pdf])

#%%
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from viz_tree.nodes import LeafNode
from sklearn.manifold import TSNE

# X_embedded = TSNE(n_components=2, learning_rate='auto',
#                   init='random', perplexity=3).fit_transform(X)

pca = PCA(n_components=2, random_state=42)
pca.fit(X)
X_embedded = pca.transform(X)

scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y)
plt.title("Color is y-value")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(*scatter.legend_elements())
plt.show()

y_leafs = np.empty_like(y)
c = 1
for node in my_viz_model.nodes:
    if isinstance(node, LeafNode):
        idx = np.where((X[:,None] == node.X).all(axis=2))[0]
        y_leafs[idx] = c
        c += 1

scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_leafs)
plt.title("Color is different leaf nodes")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(*scatter.legend_elements())
plt.show()