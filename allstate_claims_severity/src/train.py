import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import xgb
import xgboost


TRAIN_PATH = "../var/train.ordinal"
TEST_PATH = "../var/test.ordinal"
ID_PATH = "../var/id.csv"
LABEL_PATH = "../var/loss.csv"

PARAMS = {
    "eta": 0.1,
    "min_child_weight": 1,
    "max_depth": 6,
    "gamma": 1,
    "subsample": 0.8,
    "colsample_by_tree": 0.5,
    "lambda": 1,
    "silent": 0,
    "verbose_eval": True,
}

def transform(df_col, logshift=1):
    return np.log(df_col + logshift)


def inverse(df_col, logshift=1):
    return np.exp(df_col) - logshift


def train(train_path=TRAIN_PATH, label_path=LABEL_PATH, params={}):
    label = transform(pd.read_csv(label_path))
    trainset = xgboost.DMatrix(train_path, label=label)
    booster = xgboost.train(params=params, dtrain=trainset)
    return booster


def error(preds, dtrain):
    labels = dtrain.get_label()
    return "mae", mean_absolute_error(np.exp(preds), np.exp(labels))


def cv(train_path=TRAIN_PATH, label_path=LABEL_PATH):
    label = transform(pd.read_csv(label_path))
    trainset = xgboost.DMatrix(train_path, label=label)
    cv_params = {
        "num_boost_round": 1000,
        "nfold": 10,
        "early_stopping_rounds": 5,
        "seed": 2016,
    }

    cv_result = xgboost.cv(params=PARAMS, dtrain=trainset, feval=error,
                           **cv_params)
    return cv_result


def predict(booster, test_path=TEST_PATH, id_path=ID_PATH):
    testset = xgboost.DMatrix(test_path)
    ids = pd.read_csv(id_path)
    result = inverse(pd.DataFrame(booster.predict(testset), columns=["loss"]))
    pd.concat((ids, result), axis=1).to_csv("../output/submission.csv",
                                            index=False)
    print("Output written to ../output/submission.csv")
