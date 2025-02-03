import random

SVR_param = {
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "C": (0.1, 100),
    "epsilon": (0.01, 0.1),
    "degree": (1, 5),
    "gamma": ["scale", "auto"]
}

Tree_param = {
    "max_depth": (1, 3),
    "min_samples_split": (2, 3),
    "min_samples_leaf": (1, 3)
}

RandomForestRegressor_param = {
    "n_estimators": (50, 200),
    "max_depth": (1, 3),
    "min_samples_split": (2, 3),
    "min_samples_leaf": (1, 3),
    "max_features": ["auto", "sqrt", "log2"]
}

GradientBoostingRegressor_param = {
    "learning_rate": (0.01, 0.3),
    "n_estimators": (50, 200),
    "subsample": (0.5, 1.0),
    "min_samples_split": (2, 20),
    "min_samples_leaf": (1, 20),
    "max_depth": (1, 10)
}

Ridge_param = {
    "alpha": (0.1, 10)
}

Lasso_param = {
    "alpha": (0.1, 10)
}

ElasticNet_param = {
    "alpha": (0.1, 10),
    "l1_ratio": (0.1, 1.0)
}

MLPRegressor_param = {
    "hidden_layer_sizes": [(50,), (100,), (100, 50)],
    "activation": ["identity", "logistic", "tanh", "relu"],
    "solver": ["lbfgs", "sgd", "adam"],
    "alpha": (0.0001, 0.1),
    "learning_rate": ["constant", "invscaling", "adaptive"]
}

params_mapping = {
    "SVR": SVR_param,
    "Tree": Tree_param,
    "RandomForestRegressor": RandomForestRegressor_param,
    "GradientBoostingRegressor": GradientBoostingRegressor_param,
    "Ridge": Ridge_param,
    "Lasso": Lasso_param,
    "ElasticNet": ElasticNet_param,
    "MLPRegressor": MLPRegressor_param
}

def generate_param_dict(algo):
    params_range = params_mapping[algo]
    param_dict = {}
    for k, v in params_range.items():
        if isinstance(v, tuple):
            param_dict[k] = random.uniform(v[0], v[1])
            if isinstance(v[1], int) and isinstance(v[0], int):
                param_dict[k] = int(param_dict[k])
        elif isinstance(v, list):
            param_dict[k] = random.choice(v)
        else:
            raise ValueError("Unsupported parameter type.")
    return param_dict