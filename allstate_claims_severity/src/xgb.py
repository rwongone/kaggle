import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import xgboost as xgb


def cat():
    data = {
        "color": ["R", "G", "B", "R", "G", "G"],
        "sex": ["M", "F", "F", "F", "M", "M"],
        "fave_letter": ["a", "b", "c", "d", "d", "a"],
    }
    return pd.DataFrame(data=data)


def cat_test():
    data = {
        "color": ["R", "G", "B", "R", "G", "C"],
        "sex": ["M", "F", "F", "F", "M", "N"],
        "fave_letter": ["a", "b", "c", "d", "d", "f"],
    }
    return pd.DataFrame(data=data)


def csv_to_dmatrix(csv_path, id_ix=None, cat_ix=None,
                   cont_ix=None, target_ix=None,
                   cat_encoding="onehot",
                   save_to=None):
    df = pd.read_csv(csv_path)

    if id_ix:
        id_col = pd.DataFrame(df.iloc[:,id_ix])

    if cat_ix:
        cat_df = df.iloc[:,cat_ix]

    if cont_ix:
        cont_df = df.iloc[:,cont_ix]

    if target_ix:
        target_col = pd.DataFrame(df.iloc[:,target_ix])

    if save_to:
        dmat.save_binary(save_to)
        print("DMatrix saved to %s" % save_to)


def cols(df):
    return list(df.columns.values)


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

    new_train_cat = pd.DataFrame(np.column_stack(train_enc), columns=columns)
    new_test_cat = pd.DataFrame(np.column_stack(test_enc), columns=columns)
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
    new_train_cat, new_test_cat = ord_enc.iloc[:l,:], ord_enc.iloc[l:,:]
    return new_train_cat, new_test_cat
