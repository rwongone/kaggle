import pandas as pd
import xgb
import xgboost


def train(train_path):
    params = {

    }
    trainset = xgboost.DMatrix(train_path)
    booster = xgboost.train(params=params, dtrain=trainset)
    return booster
