
import random

knn_param = {
    "n_neighbors": (3,10),
    "weights": ["uniform", "distance"],
    "p": (1,10)
}

SVM_param = {
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "C": (0.1, 0.1),
    # "rp": (0.1, 0.5, 1, 5),
    # "shrink": [True, False],
    # "prob": [True, False],
    # "dfs": ["ovo", "ovr"]
}

adaboost_param = {
    "learning_rate": (0.1, 1.0),
    "n_estimators": (10, 200),
    "algorithm": ["SAMME.R", "SAMME"]
}


GBDT_param = {
    "learning_rate": (0.1, 0.3),
    "n_estimators": (50, 300),
    "subsample": (0.5, 1.0),
    "min_samples_split": (0.2, 0.8),
    "min_samples_leaf": (1, 5),
    "min_weight_fraction_leaf": (0, 0.3),
    "max_depth": (1, 10),
    "min_impurity_decrease": (0, 1)
}

RF_param = {
    "n_estimators": (100, 200),
    "criterion": ["gini"]
}

DTree_param = {
    "criterion": ["gini"],
    "splitter": ["best"],
    "random_state": (0, 0)
}

bagging_param = {
    "n_estimators": (50, 500),
    "max_samples": (1, 1),
    "max_features": (1, 1),
    "bootstrap": [True],
    "bootstrap_features": [False],
    "n_jobs": (1, 1),
    "random_state": (1, 1)
}

Bayes_param = {
    "alpha": (1.0, 1.0),
    "fit_prior": [True, False]
}

LR_param = {
    "solver": ["liblinear"],
    "C": (1.0, 1.0),
    "multi_class": ["auto"]
}

params_mapping = {
    "knn": knn_param,
    "GBDT": GBDT_param,
    "RF": RF_param,
    "SVM": SVM_param,
    "AdaBoost": adaboost_param,
    "DeTree": DTree_param,
    "bagging": bagging_param,
    "Bayes": Bayes_param,
    "LR": LR_param
}


def generate_param_dict(algo):
    params_range = params_mapping[algo]
    param_dict = {}
    for k,v in params_range.items():
        if isinstance(v, tuple):
            param_dict[k] = random.uniform(v[0], v[1])
            if isinstance(v[1], int) and isinstance(v[0], int):
                param_dict[k] = int(param_dict[k])
        elif isinstance(v, list):
            param_dict[k] = random.sample(v, 1)[0]
        else:
            raise ValueError
    return param_dict

