import numpy as np
import os.path as path
import pandas as pd
from scipy.stats import boxcox, skew
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import KFold
import traceback
from xgboost import XGBRegressor


logshift = 1
def f(loss, shift=200):
    return np.log(loss + shift)


def f_inv(logloss, shift=200):
    return np.exp(logloss) - shift


def merge_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z


def split(df_path, k=10):
    n_cat = 1176
    n_cont = 14
    dtypes = dict(zip(range(n_cat), ["uint8"] * n_cat) +
                  zip(range(n_cat, n_cat+n_cont), ["float64"] * n_cont))
    df = pd.read_csv(df_path, dtype=dtypes)
    print("Initial DataFrame info:")
    df.info()

    X = df.iloc[:,:-1]
    Y = f(df.iloc[:,-1], logshift)

    return X, Y, KFold(n_splits=k)


def read_ord(df_path, k=10, test=False):
    n_cat = 116
    n_cont = 14
    dtypes = dict([("id", "uint32")] +
                   zip(["cat%d" % (i+1) for i in range(n_cat)], ["uint16"] * n_cat) +
                   zip(["cont%d" % (i+1) for i in range(n_cont)], ["float32"] * n_cont))
    if test:
        dtypes["loss"] = "float32"

    df = pd.read_csv(df_path, dtype=dtypes)
    print("df info:")
    df.info()

    if test:
        X = df.iloc[:, 1:]
        Y = pd.DataFrame()
    else:
        X = df.iloc[:, 1:-1]
        Y = f(df.iloc[:, -1], logshift)

    ret3 = pd.DataFrame(df.iloc[:, 0]) if test else KFold(n_splits=k)
    return X, Y, ret3


def build_ensemble(regressor, kfold):
    X, Y, kf = kfold
    r, kwargs = regressor

    ensemble = []
    for i, (train, test) in enumerate(kf.split(X)):
        reg = r(**kwargs)
        X_train, X_val = X.ix[train], X.ix[test]
        Y_train, Y_val = Y[train], Y[test]

        try:
            reg.fit(X_train, Y_train, eval_metric="mae", verbose=True);
            ensemble.append(reg)
            Y_out = reg.predict(X_val)
            mae_val = mae(f_inv(Y_out, logshift), f_inv(Y_val, logshift))
            print("Fold %d mae = %.6f" % (i, mae_val))
        except MemoryError:
            print("MemoryError with %s, %s." % (reg.__class__.__name__, kwargs))
            traceback.print_exc()
        except Exception:
            print("General error.")
            traceback.print_exc()

    cv_prediction = None
    for reg in ensemble:
        Y_out = reg.predict(X_val)
        inc_pred = Y_out / kf.get_n_splits()
        if cv_prediction is None:
            cv_prediction = inc_pred
        else:
            cv_prediction = cv_prediction + inc_pred

    print("Ensemble mae = %.6f" % mae(f_inv(cv_prediction, logshift),
                                      f_inv(Y_val, logshift)))
    return ensemble


def read_test(test_path):
    n_cat = 1176
    n_cont = 14

    dtypes = dict([(0, "uint32")] + zip(range(1, n_cat+1), ["uint8"] * n_cat) +
                  zip(range(n_cat+1, n_cat+n_cont+1), ["float64"] * n_cont))
    df = pd.read_csv(test_path, dtype=dtypes)

    print("Test DataFrame info:")
    df.info()

    id_col = pd.DataFrame(df.iloc[:,0])
    X = df.iloc[:,1:]


    X.rename(columns=dict(zip([str(i) for i in xrange(1, 1191)],
                              [str(i) for i in xrange(0, 1190)])),
             inplace=True)
    del df

    return id_col, X


prediction = None
y_pred = None
inc_prediction = None
id_col = None
df = None

def predict(ensemble, id_col, X, k=5):
    prediction = None
    for i, reg in enumerate(ensemble):
        y_pred = reg.predict(X) / k
        inc_prediction = pd.DataFrame(y_pred, columns=["loss"])
        if i > 0:
            prediction = prediction + inc_prediction
        else:
            prediction = inc_prediction

    id_col = id_col.rename(columns={"0": "id"})
    df = pd.concat((id_col, prediction), axis=1)
    df["loss"] = f_inv(df["loss"], logshift)
    df.to_csv("../output/submission.csv", index=False)
    return df


xgbr = {
    "max_depth": 8,
    "reg_alpha": 1,
    "gamma": 1,
    "n_estimators": 3000,
    "min_child_weight": 30,
    "subsample": 0.9,
    "colsample_bytree": 0.7,
    "seed": 2016,
}

if __name__ == "__main__":
    k = 20
    xgb = (XGBRegressor, xgbr)
    print("Building ensemble.")
    ensemble = build_ensemble(xgb, read_ord("../input/ord_encoded.csv", k=k))
    print("Predicting.")
    X, _, id_col = read_ord("../input/ord_encoded_test.csv", test=True)
    prediction = predict(ensemble, id_col, X, k=k)
    print("Done.")
