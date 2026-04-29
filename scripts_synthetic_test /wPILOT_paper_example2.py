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
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import plotly.graph_objects as go
import plotly.io as pio

#%%
def true_density(x0, x1):
    pass

def sample_of_density(n):
    n0_high_dens = int(6/7*n)
    n0_low_dens = n - n0_high_dens
    x0_high = np.random.uniform(-3,3,n0_high_dens)
    x0_low = 3 + np.random.exponential(1,n0_low_dens)
    x0 = np.r_[x0_high, x0_low]

    n1_high_dens = int(0.8*n)
    n1_low_dens = n - n1_high_dens
    x1_high = np.random.normal(0, 1, n1_high_dens)
    x1_low = np.random.normal(4, 1, n1_low_dens)

    x1 = np.r_[x1_high, x1_low]
    np.random.shuffle(x1)
    return x0, x1

x0, x1 = sample_of_density(220)
# plt.plot(x0, x1, '.')
# plt.show()

#%%
def get_response(x0, x1):
    if x0 <= 3:
        if x1 <= 4:
            return 7 - x1 + x0
        else:
            return 1 + 0.5 * x1 + x0
    else:
        return - 4*x0 + 50

def get_response_array(x0, x1):
    y = np.empty(len(x0))
    for i in range(len(x0)):
        y[i] = get_response(x0[i], x1[i])
    return y

y = get_response_array(x0, x1) + np.random.normal(0,2,len(x0))

#%%
x0_plot = np.linspace(-3, 6, 100)
x1_plot = np.linspace(-3, 8, 100)
X0_plot, X1_plot = np.meshgrid(x0_plot, x1_plot)
Z = get_response_array(X0_plot.ravel(), X1_plot.ravel()).reshape(X0_plot.shape)

fig = go.Figure()

fig.add_surface(x=X0_plot, y=X1_plot, z=Z, opacity=0.5, colorscale="Viridis", name="true")

fig.add_scatter3d(
    x=x0,
    y=x1,
    z=y,
    mode="markers",
    marker=dict(size=3, color="black"),
    name="data"
)
fig.update_layout(
    scene=dict(
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=0.7)
    )
)

pio.renderers.default = "browser"
fig.show()


#%%
X = np.c_[x0,x1]
kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
kde.fit(X)
density = np.exp(kde.score_samples(X))

w = 1.0 / (density + 0.01)
w = w / np.mean(w)

model = PILOT(max_model_depth=2, max_depth=2)
model.fit(X, y)
modelW = PILOT(max_model_depth=2, max_depth=2)
modelW.fit(X, y, w)

plt.scatter(x0, x1, c=w, cmap='viridis')
plt.colorbar()
plt.show()

#%%
# grid predictions from model
grid = np.column_stack([X0_plot.ravel(), X1_plot.ravel()])
Z_pred  = model.predict(grid).reshape(X0_plot.shape)
Z_predW = modelW.predict(grid).reshape(X0_plot.shape)

fig = go.Figure()

# true function
fig.add_surface(x=X0_plot, y=X1_plot, z=Z, opacity=0.5, colorscale="Viridis", name="true")

# (weighted) model prediction
fig.add_surface(x=X0_plot, y=X1_plot, z=Z_pred, opacity=0.5, colorscale="Reds", name="model")
fig.add_surface(x=X0_plot, y=X1_plot, z=Z_predW, opacity=0.5, colorscale="Greens", name="model")

# data points
fig.add_scatter3d(
    x=x0,
    y=x1,
    z=y,
    mode="markers",
    marker=dict(size=3, color="black"),
    name="data"
)
fig.update_layout(
    scene=dict(
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=0.7)
    )
)

pio.renderers.default = "browser"
fig.show()


#%%
pilot_tree = modelW.model_tree

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
with open(file_directory_pdf, "wb") as f_file:
    f_file.write(pdf_bytes)
run(["open", "-a", "Preview", file_directory_pdf])

