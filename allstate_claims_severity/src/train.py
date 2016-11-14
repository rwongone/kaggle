import pandas as pd
import xgb
import xgboost


def transform(df_col, logshift=1):
    return np.log(df_col + logshift)


def inverse(df_col, logshift=1):
    return np.exp(df_col) - logshift


def train(train_path, label_path, params={}):
    label = transform(pd.read_csv(label_path))
    trainset = xgboost.DMatrix(train_path, label=label)
    booster = xgboost.train(params=params, dtrain=trainset)
    return booster


def predict(booster, test_path, id_path):
    testset = xgboost.DMatrix(test_path)
    ids = pd.read_csv(id_path)
    result = inverse(pd.DataFrame(booster.predict(testset), columns=["loss"]))
    pd.concat((ids, result), axis=1).to_csv("../output/submission.csv",
                                            index=False)
    print("Output written to ../output/submission.csv")
