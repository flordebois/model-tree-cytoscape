import pybaobabdt
import pandas as pd
from scipy.io import arff
from sklearn.tree import DecisionTreeClassifier

#%%
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from colour import Color
import matplotlib.pyplot as plt
import numpy as np

data = arff.loadarff('/Users/flor/Pycharm/PILOT-VIS/scripts_boabab/vehicle.arff')
df   = pd.DataFrame(data[0])
y = list(df['class'])
features = list(df.columns)

                   
features.remove('class')
X = df.loc[:, features]

clf = DecisionTreeClassifier().fit(X, y)
#%%
ax = pybaobabdt.drawTree(clf, size=10, dpi=72, features=features, colormap='Spectral')
#%%
ax.get_figure().savefig('/Users/flor/Pycharm/PILOT-VIS/scripts_boabab/tree_example.pdf')
#%%
from sklearn.tree import export_graphviz

export_graphviz(
    clf,
    out_file='/Users/flor/Pycharm/PILOT-VIS/scripts_boabab/model.dot',
    feature_names=features,
    rounded=True,
    filled=False)

#!dot -Tpng model.dot -o vehicle_dt.png
#%%
from IPython.display import Image
Image(filename='vehicle_dt.png')