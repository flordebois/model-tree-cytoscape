print("importing PILOT...")
from pilot.pilot import PILOT
from pilot.viz_tree import VizTree

print("import done")

import numpy as np
import matplotlib.pyplot as plt
from pmlb import fetch_data
from graphviz import Source
# Use 'TkAgg' for external window
import matplotlib
matplotlib.use('TkAgg')

#%%
np.random.seed(0)
n = 2000

A = np.random.randn(n, 2)
# orthonormal basis via QR
Q, _ = np.linalg.qr(A)
u = Q[:, 0]
v = Q[:, 1]

# build X1 and X2 mostly aligned with u but with small opposite components along v
a = 0.4
X1 = u + a * v
X2 = u - a * v

# response depends mainly on v (so each predictor explains only a small part of it)
y = 4.0 * v + 0.05 * np.random.randn(n)   # strong signal along v, small observation noise

# combine into matrix X (n x 2) and make sure shapes are numpy ndarrays
X = np.column_stack((X1, X2))
y = y.copy()

feature_names = ["X1", "X2"]
target_name = "Target"

#%%
pilot_model = PILOT()
pilot_model.fit(X, y)
pilot_tree = pilot_model.model_tree

#%%
my_viz_model = VizTree(tree_model = pilot_tree,
                       X_train=X,
                       y_train=y,
                       feature_names=feature_names,
                       target_name=target_name)

#%%
pilot_dot = my_viz_model.get_dot(combine_lin=True)
Source(pilot_dot).render(f"Output/pilot_graph_test_corr.gv", format="pdf", view=True)

#%%
plt.plot(A[:,0],A[:,1], '.')
plt.xlabel('A0')
plt.ylabel('A1')
plt.title(f'Corr={round(np.corrcoef(A[:,0],A[:,1])[0,1],4)}')
plt.show()

#%%
plt.plot(u,v, '.')
plt.xlabel('u')
plt.ylabel('v')
plt.title(f'Corr={round(np.corrcoef(u,v)[0,1],4)}')
plt.show()

#%%
plt.plot(X2, y, '.')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title(f'Corr={round(np.corrcoef(X1,X2)[0,1],4)}\nTheo Corr={round((1-a**2)/(1+a**2),4)}')
plt.show()

#%%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

x_range = np.linspace(min(X1), max(X1), 10)
y_range = np.linspace(min(X2), max(X2), 10)
X_plane, Y_plane = np.meshgrid(x_range, y_range)
Z_plane = pilot_tree.lm_l[0] * (X_plane if pilot_tree.pivot[0] == 0 else Y_plane)
Z_plane2 = -2/a * (X_plane if pilot_tree.pivot[0] == 0 else Y_plane)
Z_plane3 = 2/a*(X_plane - Y_plane)

ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.5, color='red')
ax.plot_surface(X_plane, Y_plane, Z_plane2, alpha=0.5, color='blue')
ax.plot_surface(X_plane, Y_plane, Z_plane3, alpha=0.5, color='green')
ax.scatter(X1, X2, y)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
ax.set_zlim(min(y), max(y))
plt.show()

#%%
tree_i = pilot_tree
coef1 = 0
coef2 = 0
while tree_i.left != None:
    if tree_i.pivot[0] == 0:
        coef1 += tree_i.lm_l[0]
    else:
        coef2 += tree_i.lm_l[0]
    print(f"Variable X_{tree_i.pivot[0]+1}")
    print(f"Coef={np.round(tree_i.lm_l[0],3)}, intercept={np.round(tree_i.lm_l[1],3)}")
    tree_i = tree_i.left
print(f"Total Coef X1={np.round(coef1,2)}, Coef X2={np.round(coef2,2)}")
print(f"Total Coef X1 {np.round(2/a,2)}")