import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def split(df_path, test_size=0.1, seed=0):
    df = pd.read_csv(df_path)
    print("Initial DataFrame info:")
    df.info()

    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1]

    return train_test_split(X, Y, test_size=test_size,
                            random_state=seed)


def mae(Y_out, Y_val):
    assert len(Y_out) == len(Y_val)
    return np.sum(np.abs(Y_out - Y_val)) / len(Y_out)


def try_models(regressors, tt_split):
    X_train, X_val, Y_train, Y_val = tt_split
    print("X_train info:")
    X_train.info()
    del tt_split
    for r, kwargs in regressors:
        reg = r(**kwargs)
        reg.fit(X_train, Y_train)
        Y_out = reg.predict(X_val)
        print(mae(Y_out, Y_val))


regressors = [(GradientBoostingRegressor, {"loss": "ls", "learning_rate": 0.1, "n_estimators": 50, "max_depth": 3})]

try_models(regressors, split("../encoded.csv"))

