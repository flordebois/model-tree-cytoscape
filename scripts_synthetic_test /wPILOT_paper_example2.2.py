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

#%%
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

#%%
def get_response(x0, x1):
    if x0 <= 3:
        if x1 <= 4:
            return 7 - x1 + x0
        else:
            return 1 + 0.5 * x1 + x0
    else:
        if x0 <= 6.1:
            return - 4*x0 + 50
        else:
            return 30

def get_response_array(x0, x1):
    y = np.empty(len(x0))
    for i in range(len(x0)):
        y[i] = get_response(x0[i], x1[i])
    return y

#%%
def run_pilot_experiment(n=220, noise=2, weighting=0.01):
    x0, x1 = sample_of_density(n)
    y = get_response_array(x0, x1) + np.random.normal(0, noise, len(x0))

    x0_plot = np.linspace(-3, 8, 100)
    x1_plot = np.linspace(-3, 8, 100)
    X0_plot, X1_plot = np.meshgrid(x0_plot, x1_plot)
    Z = get_response_array(X0_plot.ravel(), X1_plot.ravel()).reshape(X0_plot.shape)

    # fig = go.Figure()
    # fig.add_surface(x=X0_plot, y=X1_plot, z=Z, opacity=0.5, colorscale="Viridis", name="true")
    # fig.add_scatter3d(x=x0, y=x1, z=y, mode="markers", marker=dict(size=3, color="black"), name="data")
    # fig.update_layout(scene=dict(aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.7)))
    # fig.show()

    X = np.c_[x0, x1]
    kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
    kde.fit(X)
    density = np.exp(kde.score_samples(X))

    w = 1.0 / (density + weighting)
    w = w / np.mean(w)

    model = PILOT(max_model_depth=2, max_depth=2)
    model.fit(X, y)
    modelW = PILOT(max_model_depth=2, max_depth=2)
    modelW.fit(X, y, w)

    plt.scatter(x0, x1, c=w, cmap='viridis')
    plt.colorbar()
    plt.show()

    grid = np.column_stack([X0_plot.ravel(), X1_plot.ravel()])
    Z_pred  = model.predict(grid).reshape(X0_plot.shape)
    Z_predW = modelW.predict(grid).reshape(X0_plot.shape)

    fig = go.Figure()
    fig.add_surface(x=X0_plot, y=X1_plot, z=Z, opacity=0.5, colorscale="Viridis", name="true")
    fig.add_surface(x=X0_plot, y=X1_plot, z=Z_pred, opacity=0.5, colorscale="Reds", name="model")
    fig.add_surface(x=X0_plot, y=X1_plot, z=Z_predW, opacity=0.5, colorscale="Greens", name="model")
    fig.add_scatter3d(x=x0, y=x1, z=y, mode="markers", marker=dict(size=3, color="black"), name="data")
    fig.update_layout(scene=dict(aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.7)))
    fig.show()

    # if output_dir:
    #     file_name = 'synthetic_' + true_name.lower().replace(' ', '_') + '.pdf'
    #     plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
    plt.show()

    print("Pilot tree:")
    model.print_tree()
    print("Weigthed Pilot tree:")
    modelW.print_tree()
    return X, y, model, modelW

#%%
np.random.seed(45)
n = 220
noise = 2
#weighting = 0.01
weighting = 0.001
X, y, model, modelW = run_pilot_experiment(n, noise, weighting)

#%%
output_directory = "/Users/flor/Pycharm/PILOT/output/synthetic_example"

def get_plot_data(model, X, y):
    model_tree = model.model_tree
    left_mask = X[:,model_tree.pivot[0]] <= model_tree.pivot[1]
    X_left = X[left_mask]
    y_left = y[left_mask]
    X_right = X[~left_mask]
    y_right = y[~left_mask]
    w_left = model.w.flatten()[left_mask]
    w_right = model.w.flatten()[~left_mask]

    def left_res(X, y):
        return y - (model_tree.lm_l[0]*X + model_tree.lm_l[1])
    def right_res(X, y):
        return y - (model_tree.lm_r[0]*X + model_tree.lm_r[1])

    # y_left = left_res(X_left[:,model_tree.pivot[0]], y_left)
    # y_right = right_res(X_right[:,model_tree.pivot[0]], y_right)

    x_plot_left = np.linspace(-3, 6, 1000)
    x_plot_right = np.linspace(3.01, 8, 500)

    y_true_left = get_response_array(np.zeros(1000), x_plot_left)
    #y_true_left = left_res(np.zeros(1000), y_true_left)
    y_true_right = get_response_array(x_plot_right, np.zeros(500))
    #y_true_right = right_res(x_plot_right, y_true_right)

    y_pred_left = model.predict(np.column_stack([np.zeros(1000),x_plot_left]))
    #y_pred_left = left_res(np.zeros(1000), y_pred_left)
    y_pred_right = model.predict(np.column_stack([x_plot_right, np.zeros(500)]))
    #y_pred_right = right_res(x_plot_right, y_pred_right)

    l_list = [X_left[:,1], y_left, x_plot_left, y_true_left, y_pred_left, w_left]
    r_list = [X_right[:,0], y_right, x_plot_right, y_true_right, y_pred_right, w_right]
    return l_list, r_list


left_list, right_list = get_plot_data(model, X, y)
wleft_list, wright_list = get_plot_data(modelW, X, y)


# plot left subtree
x_label_string = r"x^{(" + str(2) + r")}"
fig, ax = plt.subplots(figsize=(5.5, 3.55))
p1 = ax.scatter(left_list[0], left_list[1], s=18, color='slategrey',
                label='Training data', alpha=0.6, edgecolor='none')

scaled_weights_left = (wleft_list[5] - np.median(wleft_list[5]))*15 + 15
p2 = ax.scatter(wleft_list[0], wleft_list[1], s=scaled_weights_left,
               color='C0', label='Weighted data', alpha=0.5, edgecolor='none')

l1, = ax.plot(left_list[2], left_list[3], 'k-', linewidth=2, label='True function')
#l2, = ax.plot(wleft_list[2], wleft_list[3], 'k--', linewidth=2, label='True function')

lp1, = ax.plot(left_list[2], left_list[4], color='red',
               linestyle=(0, (3, 1, 1, 1)), linewidth=2, label='PILOT')
lp2, = ax.plot(wleft_list[2], wleft_list[4], color='green',
               linestyle=(0, (5, 1)), linewidth=2, label='Weighted PILOT')

ax.set_xlabel(f'${x_label_string}$')
#ax.set_ylabel('Residuals')
ax.set_ylabel('$y$')
ax.set_title('Left child of the root node')

ax.legend(loc='upper left', bbox_to_anchor=(0.555,1) )
#legend1 = ax.legend(handles=[p1, l1, lp1], loc='upper right', title='PILOT')
#ax.add_artist(legend1)
#ax.legend(handles=[p2, l2, lp2], loc='lower left', title='Weighted PILOT')
plt.tight_layout()
fig.savefig(output_directory + "/left_child_synthetic_example_v2.pdf", bbox_inches='tight')
plt.show()

# # plot right subtree
x_label_string = r"x^{(" + str(1) + r")}"
fig, ax = plt.subplots(figsize=(5.5, 3.55))
p1 = ax.scatter(right_list[0], right_list[1], s=18, color='slategrey',
                label='Training data', alpha=0.6, edgecolor='none')

scaled_weights_right = wright_list[5]*15 + 4
p2 = ax.scatter(wright_list[0], wright_list[1], s=scaled_weights_right,
                color='C0', label='Weighted data', alpha=0.5, edgecolor='none')

l1, = ax.plot(right_list[2], right_list[3], 'k-', linewidth=2, label='True function')
#l2, = ax.plot(wright_list[2], wright_list[3], 'k--', linewidth=2, label='True function')

lp1, = ax.plot(right_list[2], right_list[4], color='red',
               linestyle=(0, (3, 1, 1, 1)), linewidth=2, label='PILOT')
lp2, = ax.plot(wright_list[2], wright_list[4], color='green',
               linestyle=(0, (5, 1)), linewidth=2, label='Weighted PILOT')

ax.set_xlabel(f'${x_label_string}$')
#ax.set_ylabel('Residuals')
ax.set_ylabel('$y$')
ax.set_title('Right child of the root node')

ax.legend(loc='upper right')
#legend1 = ax.legend(handles=[p1, l1, lp1], loc='upper center', title='PILOT')
#ax.add_artist(legend1)
#ax.legend(handles=[p2, l2, lp2], loc='upper right', title='Weighted PILOT')
plt.tight_layout()
plt.savefig(output_directory + "/right_child_synthetic_example_v2.pdf", bbox_inches='tight')
plt.show()

#%%
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
                           use_regplot=False,
                           print_model = False,
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


