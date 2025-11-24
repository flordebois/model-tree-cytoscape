import numpy as np

def synthetic_dataset(name, n=1000, noise=1, seed=42):
    if name == "toy":
        return make_toy_dataset(n, noise=noise, seed=seed)
    elif name == "medium":
        return make_medium_dataset(n, noise=noise, seed=seed)
    elif name == "hard":
        return make_hard_dataset(n, seed=seed)
    else:
        raise NotImplementedError


###################### Made with ChatGPT ######################
def make_toy_dataset(n=300, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    X1 = rng.uniform(-2, 2, n)
    X2 = rng.uniform(-2, 2, n)
    noise_term = rng.normal(0, noise, n)

    y = 3*X1 + 2*(X2**2) + noise_term

    # add 3 noise features
    X3 = rng.normal(0, 1, n)
    X4 = rng.normal(0, 1, n)
    X5 = rng.normal(0, 1, n)

    X = np.c_[X1, X2, X3, X4, X5]
    return X,y

def make_medium_dataset(n=3000, noise=1.0, seed=1):
    rng = np.random.default_rng(seed)

    # Correlated block
    base = rng.normal(0, 1, (n, 3))
    X1 = base[:, 0]
    X2 = base[:, 1] + 0.3*base[:, 0]
    X3 = base[:, 2] + 0.5*base[:, 1]

    X4 = rng.uniform(-3, 3, n)
    X5 = rng.normal(2, 1, n)

    # Irrelevant features
    X_noise = rng.normal(0, 1, (n, 5))

    noise_term = rng.normal(0, noise, n)

    y = 4*np.sin(X1) + X2*X3 + 0.5*(X4**2) + 2*X5 + noise_term

    X = np.c_[X1, X2, X3, X4, X5, X_noise]
    return X,y

def make_hard_dataset(n=8000, seed=2):
    rng = np.random.default_rng(seed)

    # True features
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    X3 = rng.normal(0, 2, n)
    X4 = rng.uniform(-3, 3, n)
    X5 = rng.uniform(-1, 1, n)
    X6 = rng.normal(0, 1, n)

    # Irrelevant features
    X_noise = rng.normal(0, 1, (n, 14))

    # Heteroscedastic noise
    noise = rng.normal(0, 1 + 0.5*np.abs(X4), n)

    y = (
        3*(X1 > 0)*X2 +
        5*np.sqrt(np.abs(X3)) -
        2*(X4**2) +
        4*np.sin(X5 * X6) +
        noise
    )

    X = np.c_[X1, X2, X3, X4, X5, X6, X_noise]
    return X, y

