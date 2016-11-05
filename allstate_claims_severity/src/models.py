import numpy as np
import os.path as path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
import traceback
from xgboost import XGBRegressor


def merge_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z


def split(df_path, test_size=0.1, seed=0):
    n_cat = 1175
    dtypes = dict(zip(range(n_cat), ["uint8"] * n_cat) +
                  zip(range(n_cat, n_cat+15), ["float64"] * 15))
    df = pd.read_csv(df_path, dtype=dtypes)
    print("Initial DataFrame info:")
    df.info()

    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1]
    return train_test_split(X, Y, test_size=test_size,
                            random_state=seed)


def kfold_split(df_path, k=10, seed=0):
    n_cat = 1175
    dtypes = dict(zip(range(n_cat), ["uint8"] * n_cat) +
                  zip(range(n_cat, n_cat+15), ["float64"] * 15))
    df = pd.read_csv(df_path, dtype=dtypes)
    print("Initial DataFrame info:")
    df.info()

    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1]
    return KFold(n_splits=k)


def mae(Y_out, Y_val):
    assert len(Y_out) == len(Y_val)
    return np.sum(np.abs(Y_out - Y_val)) / len(Y_out)


def try_models(regressors, tt_split):
    X_train, X_val, Y_train, Y_val = tt_split
    del tt_split
    out = []
    for r, kwargs in regressors:
        reg = r(**kwargs)
        try:
            if r is XGBRegressor:
                reg.fit(X_train, Y_train, eval_metric="mae", verbose=True)
            else:
                reg.fit(X_train, Y_train)
            Y_out = reg.predict(X_val)
            mae_val = mae(np.expm1(Y_out), np.expm1(Y_val))
            del Y_out
            output = (reg.__class__.__name__, kwargs, mae_val)
            print(output)
            out.append(output)
            del reg
        except MemoryError:
            print("MemoryError with %s, %s." % (reg.__class__.__name__, kwargs))
            traceback.print_exc()

    return out

seed = 2016

rfr = {
    "n_jobs" : -1,
    "verbose": 3,
    "random_state": seed,
}

etr = {
    "n_jobs": -1,
    "verbose": 3,
    "random_state": seed,
}

xgbr = {
    "max_depth": 6,
    "alpha": 1,
    "gamma": 1
}

gbr = {
    "loss": "ls",
    "learning_rate": 0.1,
    "max_depth": 3,
    "verbose": 3,
    "random_state": seed,
}

regressors = [
    (XGBRegressor, merge_dicts(xgbr, { "n_estimators": 100 })),
    # (GradientBoostingRegressor, merge_dicts(gbr, { "n_estimators": 100, })),
]

if __name__ == "__main__":
    output = try_models(regressors, split("../encoded.csv"), seed=seed)
    print(output)
