import numpy as np
from pilot.pilot import PILOT, DEFAULT_DF_SETTINGS
from sklearn.base import BaseEstimator

class coPILOT(BaseEstimator):

    def __init__(self,
                 normalize_w = True,
                 max_n_estimators=50,
                 max_depth=12,
                 max_model_depth=30,
                 split_criterion="BIC",
                 min_sample_split=10,
                 min_sample_leaf=5,
                 step_size=1,
                 random_state=42,
                 truncation_factor: int = 3,
                 rel_tolerance: float = 0,
                 df_settings: dict[str, int] | None = None,
                 regression_nodes: list[str] | None = None,
                 min_unique_values_regression: float = 5,
                 alpha: float = 1
                 ) -> None:
        self.normalize_w = normalize_w
        self.max_n_estimators = max_n_estimators

        # Pilot parameters
        self.max_depth = max_depth
        self.max_model_depth = max_model_depth
        self.split_criterion = split_criterion
        self.min_sample_split = min_sample_split
        self.min_sample_leaf = min_sample_leaf
        self.step_size = step_size
        self.random_state = random_state
        self.truncation_factor = truncation_factor
        self.rel_tolerance = rel_tolerance
        self.df_settings = (
            df_settings
            if df_settings is not None
            else dict(zip(DEFAULT_DF_SETTINGS.keys(),
                          (1 + alpha * (np.array(list(DEFAULT_DF_SETTINGS.values())) - 1)).tolist()
                          ))
        )
        self.regression_nodes = regression_nodes
        self.min_unique_values_regression = min_unique_values_regression
        self.alpha = alpha

        # Parameters for fitting
        self.n_estimators = 0
        self.X = None
        self.y = None
        self.pilot_trees: list[PILOT] = []
        self.losses: list[float] = []
        self.betas: list[float] = []

    def fit(self, X, y, categorical = np.array([-1]), verbose=False, stop_early=True):
        w = np.ones(X.shape[0])
        for i in range(self.max_n_estimators):
            if verbose:
                print(f"Fitting Pilot tree number {i+1} of maximum {self.max_n_estimators}")
            pilot_model = PILOT(self.max_depth,
                                self.max_model_depth,
                                self.split_criterion,
                                self.min_sample_split,
                                self.min_sample_leaf,
                                self.step_size,
                                self.random_state,
                                self.truncation_factor,
                                self.rel_tolerance,
                                self.df_settings,
                                self.regression_nodes,
                                self.min_unique_values_regression)
            pilot_model.fit(X, y, w, normalize_w=self.normalize_w, categorical=categorical)

            if i!= 0 and stop_early and _only_lin_con(pilot_model.model_tree):
                if verbose:
                    print(f"Boosting stopped: Pilot tree number {i + 1} (not included) had only linear or constant nodes.")
                break
            self.pilot_trees.append(pilot_model)

            error = (y - pilot_model.predict(X))**2
            loss = error/error.max()
            self.losses.append(loss)
            weighted_loss = np.sum(loss * w)/np.sum(w)
            beta = weighted_loss/(1-weighted_loss)
            self.betas.append(beta)

            # Prepare next iteration
            w = w * beta**(1-loss)
            self.n_estimators += 1

    def predict(self, X, limit = None, method = "Median"):
        if limit is None:
            limit = self.n_estimators
        if method == "Median":
            weights = np.array([np.log(1/np.array(beta)) for beta in self.betas[:limit]])
            predictions = np.array([pilot_tree.predict(X) for pilot_tree in self.pilot_trees[:limit]]).T

            sorted_index = np.argsort(predictions, axis=1)
            weights_cdf = np.cumsum(weights[sorted_index], axis=1)

            threshold = 0.5 * weights_cdf[:, -1][:, np.newaxis]
            median_idx = (weights_cdf >= threshold).argmax(axis=1)

            median_prediction_idx = sorted_index[np.arange(X.shape[0]),median_idx]
            return predictions[np.arange(X.shape[0]), median_prediction_idx]
        elif method == "Average":
            predictions = np.array([pilot_tree.predict(X) for pilot_tree in self.pilot_trees[:limit]]).T
            return np.average(predictions, axis=1)
        else:
            raise ValueError(f"Unknown predict method: {method}")

def _only_lin_con(model_tree):
    if (model_tree.node == 'blin' or
        model_tree.node == 'plin' or
        model_tree.node == 'pcon' or
        model_tree.node == 'pconc'):
        return False
    else:
        if model_tree.left is None:
            return True
        else:
            return _only_lin_con(model_tree.left)