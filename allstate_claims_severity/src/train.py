import numpy as np
import pandas as pd
import xgb
import xgboost


TRAIN_PATH = "../var/train.onehot"
TEST_PATH = "../var/test.onehot"
ID_PATH = "../var/id.csv"
LABEL_PATH = "../var/loss.csv"

def transform(df_col, logshift=1):
    return np.log(df_col + logshift)


def inverse(df_col, logshift=1):
    return np.exp(df_col) - logshift


def train(train_path=TRAIN_PATH, label_path=LABEL_PATH, params={}):
    label = transform(pd.read_csv(label_path))
    trainset = xgboost.DMatrix(train_path, label=label)
    booster = xgboost.train(params=params, dtrain=trainset)
    return booster


def predict(booster, test_path=TEST_PATH, id_path=ID_PATH):
    testset = xgboost.DMatrix(test_path)
    ids = pd.read_csv(id_path)
    result = inverse(pd.DataFrame(booster.predict(testset), columns=["loss"]))
    pd.concat((ids, result), axis=1).to_csv("../output/submission.csv",
                                            index=False)
    print("Output written to ../output/submission.csv")
