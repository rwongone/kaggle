import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import xgboost


def cols(df):
    return list(df.columns.values)


def A():
    data = {
        "color": ["R", "G", "B", "R", "G", "G"],
        "sex": ["M", "F", "F", "F", "M", "M"],
        "fave_letter": ["a", "b", "c", "d", "d", "a"],
        "numbers": [1, 2, 3, 4, 1, 2],
    }
    return pd.DataFrame(data=data)


def test():
    def A():
        data = {
            "color": ["R", "G", "B", "R", "G", "G"],
            "sex": ["M", "F", "F", "F", "M", "M"],
            "fave_letter": ["a", "b", "c", "d", "d", "a"],
            "numbers": [1, 2, 3, 4, 1, 2],
        }
        return pd.DataFrame(data=data)

    def B():
        data = {
            "color": ["R", "G", "B", "R", "G", "C"],
            "sex": ["M", "F", "F", "F", "M", "N"],
            "fave_letter": ["a", "b", "c", "d", "d", "f"],
            "numbers": [6, 4, 8, 7, 2, 1],
        }
        return pd.DataFrame(data=data)

    return (cat_encode(A(), B(), (0, 2), mode="onehot"),
            cat_encode(A(), B(), (0, 2), mode="ordinal"))


def df_to_dmatrix(df, save_to=None):
    """
    Convert a DataFrame to a DMatrix.
    """
    to_replace = ["int64", "uint8", "float64"]
    replace_with = ["int", "int", "float"]
    f_types = df.dtypes.replace(to_replace, replace_with)
    dmat = xgboost.DMatrix(data=df.values, feature_names=cols(df),
                           feature_types=f_types)
    if save_to:
        dmat.save_binary(save_to)
        print("DMatrix saved to %s" % save_to)
    return dmat


def onehot(train_cat, test_cat):
    """Return new one-hot encoded versions of train and test sets.

    Operates on pd.DataFrame.

    Assumes:
        All columns are categorical.
        set(train_cat.columns.tolist()) == set(test_cat.columns.tolist())
    """
    train_enc = []
    test_enc = []
    columns = []
    for c in cols(train_cat):
        train = train_cat.loc[:,c]
        test = test_cat.loc[:,c]
        labels = list(set(train.tolist()) | set(test.tolist()))

        l_encoder = LabelEncoder()
        l_encoder.fit(labels)
        oh_encoder = OneHotEncoder(sparse=False, n_values=len(labels))

        train_ftr = l_encoder.transform(train).reshape(len(train), 1)
        train_enc.append(oh_encoder.fit_transform(train_ftr))
        test_ftr = l_encoder.transform(test).reshape(len(test), 1)
        test_enc.append(oh_encoder.fit_transform(test_ftr))

        col_tmp = list(l_encoder.inverse_transform(range(len(labels))))
        columns = columns + ["%s_%s" % (c, i) for i in col_tmp]

    new_train_cat = pd.DataFrame(np.column_stack(train_enc), columns=columns,
                                 dtype=np.uint8)
    new_test_cat = pd.DataFrame(np.column_stack(test_enc), columns=columns,
                                dtype=np.uint8)
    return new_train_cat, new_test_cat


def ordinal(train_cat, test_cat):
    """Return new ordinal-encoded versions of train and test sets.

    Operates on pd.DataFrame.

    Assumes:
        All columns are categorical.
        set(train_cat.columns.tolist()) == set(test_cat.columns.tolist())
    """
    l = len(train_cat)
    f = lambda col: np.unique(col, return_inverse=True)[1]
    combined = pd.concat([train_cat, test_cat])
    ord_enc= combined.apply(f, axis=0)
    new_train_cat = ord_enc.iloc[:l,:].astype(np.uint8)
    new_test_cat = ord_enc.iloc[l:,:].astype(np.uint8)
    return new_train_cat, new_test_cat


def cat_encode(train, test, cat_ix, mode="onehot"):
    if type(cat_ix) is tuple:
        a, b = cat_ix
        train_cat = train.iloc[:,a:b]
        test_cat = test.iloc[:,a:b]
    elif cat_ix is not None:
        train_cat = train.iloc[:,cat_ix]
        test_cat = test.iloc[:,cat_ix]
    else:
        return train, test

    if mode == "onehot":
        train_cat_enc, test_cat_enc = onehot(train_cat, test_cat)
    elif mode == "ordinal":
        train_cat_enc, test_cat_enc = ordinal(train_cat, test_cat)
    else:
        print("mode can be one of [\"onehot\", \"ordinal\"].")
        return

    if type(cat_ix) is tuple:
        train_df = pd.concat((train.iloc[:,:a],
                              train_cat_enc, train.iloc[:,b:]), axis=1)
        test_df = pd.concat((test.iloc[:,:a], test_cat_enc, test.iloc[:,b:]),
                            axis=1)
    else:
        train_ix = [c for c in train.columns if c not in train_cat_enc.columns]
        train_df = pd.concat((train.iloc[:,train_ix], train_cat_enc), axis=1)
        test_ix = [c for c in test.columns if c not in test_cat_enc.columns]
        test_df= pd.concat((test.iloc[:,test_ix], test_cat_enc), axis=1)

    return train_df, test_df


def encode(mode="onehot"):
    modes = ["onehot", "ordinal"]
    if mode not in modes:
        return

    train_path = "../input/train.csv"
    test_path = "../input/test.csv"
    id_ix = 0
    cat_ix = (1, 117)
    cont_ix = (117, 131)
    target_ix = 132

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train_df, test_df = cat_encode(train, test, cat_ix, mode)
    del train
    del test

    train_df.to_csv("../input/train_enc.csv", index=False)
    test_df.to_csv("../input/test_enc.csv", index=False)
    del test_df
    train_dmat = df_to_dmatrix(train_df, save_to="../var/train.%s" % mode)
    del train_df
    test_df = pd.read_csv("../input/test_enc.csv")
    test_dmat = df_to_dmatrix(test_df, save_to="../var/test.%s" % mode)
    del test_df
    return train_dmat, test_dmat
