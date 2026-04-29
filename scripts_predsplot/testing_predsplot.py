import numpy as np
from pmlb import fetch_data
from viz_tree.predsplot import predsplot
from sklearn.linear_model import LinearRegression
from datetime import datetime
import os
from subprocess import run

#%%
dataset_name = "547_no2" # "547_no2" "294_satellite_image" #"658_fri_c3_250_25"
data = fetch_data(dataset_name)
X = np.array(data.iloc[:, :-1])
y = np.array(data.iloc[:, -1])
feature_names = data.iloc[:, :-1].columns.tolist()
target_name = "Target"

#%%
lin_model = LinearRegression()
lin_model.fit(X, y)
y_hat = lin_model.predict(X)
coefficients = lin_model.coef_
#coefficients[-5:-1] = 0
intercept = lin_model.intercept_

#%%
directory = "/Users/flor/Pycharm/PILOT-VIS/scripts_predsplot/output/"
id_results = datetime.now().strftime('%d-%m-%y_%H-%M-%S')
map_directory = os.path.join(directory, dataset_name)
os.makedirs(map_directory, exist_ok=True)
file_directory =  os.path.join(map_directory, f"predsplot_{dataset_name}_{id_results}.pdf")

point = X[0,:].copy()
point[3] = -10
point[1] = 22
point[5] = 100
#point[6] = 20000
predsplot(X, coefficients,y_hat,
          n_max=4,
          fig_size=(10,6),
          truncate_total_pred=True,
          variable_tick_width = True,
          file_directory=file_directory,
          highlight_x=None,
          intercept=None,
          staircase=True,
          feature_names=feature_names,
          )
run(["open", "-a", "Preview", file_directory])