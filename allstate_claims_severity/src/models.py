import numpy as np
import os.path as path
import pandas as pd
from scipy.stats import boxcox, skew
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import KFold
import traceback
from xgboost import XGBRegressor


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
    Y = df.iloc[:,-1]

    return X, Y, KFold(n_splits=k).split(X)


def build_ensemble(regressor, kfold):
    X, Y, kf = kfold
    r, kwargs = regressor

    ensemble = []
    cv_prediction = None
    for i, (train, test) in enumerate(kf):
        reg = r(**kwargs)
        X_train, X_val = X.ix[train], X.ix[test]
        Y_train, Y_val = Y[train], Y[test]

        try:
            reg.fit(X_train, Y_train, eval_metric="mae", verbose=True);
            ensemble.append(reg)
            Y_out = reg.predict(X_val)

            if cv_prediction:
                cv_prediction = cv_prediction + np.expm1(Y_out) / len(kf)
            else:
                cv_prediction = np.expm1(Y_out) / len(kf)

            mae_val = mae(np.expm1(Y_out), np.expm1(Y_val))
            print("Fold %d mae = %.6f" % (i, mae_val))
        except MemoryError:
            print("MemoryError with %s, %s." % (reg.__class__.__name__, kwargs))
            traceback.print_exc()
        except Exception:
            print("General error.")
            traceback.print_exc()

        print("Ensemble mae = %.6f" % mae(cv_prediction, np.expm1(Y_val)))

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


def predict(ensemble, id_col, X, k=5):
    prediction = None
    for i, reg in enumerate(ensemble):
        y_pred = reg.predict(X)
        inc_prediction = pd.DataFrame(np.expm1(y_pred / k), columns=["loss"])
        if i > 0:
            prediction = prediction + inc_prediction
        else:
            prediction = inc_prediction

    id_col = id_col.rename(columns={"0": "id"})
    df = pd.concat((id_col, prediction), axis=1)
    df.to_csv("../submission.csv", index=False)
    return df


xgbr = {
    "max_depth": 6,
    "reg_alpha": 0.1,
    "gamma": 1,
    "n_estimators": 100,
    "seed": 2016,
}

if __name__ == "__main__":
    k = 5
    xgb = (XGBRegressor, xgbr)
    print("Building ensemble.")
    ensemble = build_ensemble(xgb, split("../encoded.csv", k=k))
    print("Predicting.")
    prediction = predict(ensemble, *(read_test("../encoded_test.csv")), k=k)
    print("Done.")
