import pickle

import sys

from matplotlib.pyplot import tight_layout

import pilot.pilot as new_module
sys.modules['pilot.Pilot'] = new_module

import pilot.tree as new_module2
sys.modules['pilot.Tree'] = new_module2

with open('/Users/flor/Pycharm/PILOT/Results Benchmark CS/Results_29-10-25_19-01-50/slump_test_pilot_trees.pkl', 'rb') as inp:
    model = pickle.load(inp)
    model_ideal = pickle.load(inp)
    model_NW = pickle.load(inp)

#%%
from pilot.viz_tree import VizTree
from graphviz import Source

viz_model  = model_NW
my_viz_model = VizTree(tree_model = viz_model.model_tree,
                       X_train=viz_model.X,
                       y_train=viz_model.y,
                       feature_names=None,
                       target_name=None)

#%%
pilot_dot = my_viz_model.get_dot(combine_lin=False)
Source(pilot_dot).render(f"viz_tree/output/slump_test_case/pilot_graph_slump_test.gv", format="pdf", view=True)

#%%
import matplotlib.pyplot as plt
plt.figure(figsize = (3,3))
plt.plot(model.w, model_ideal.w, '.')
plt.plot([0,9],[0,9], 'k', linewidth=0.6)
plt.xlabel('Estimated weight (logistic)')
plt.ylabel('Ideal weight')
plt.tight_layout()

plt.savefig("viz_tree/output/slump_test_case/pilot_weights_graph_slump_test.pdf", bbox_inches='tight')
plt.show()
