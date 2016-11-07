import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def encode(csv_path, n_cat, n_cont):
    orig_df = pd.read_csv(csv_path)
    test_df = pd.read_csv("../input/test.csv")
    cat_df = orig_df.iloc[:, 1:n_cat+1]
    cat_test_df = test_df.iloc[:, 1:n_cat+1]
    cont_df = orig_df.iloc[:, n_cat+1:-1]
    target = pd.DataFrame(orig_df.iloc[:, -1])

    del test_df

    categories = []
    for i in xrange(n_cat):
        srs = cat_df.iloc[:,i]
        # Need to know full set of labels for a column. Take from test set.
        labels = list(set(srs.unique().tolist()) |
                      set(cat_test_df.iloc[:, i].unique().tolist()))
        lab_enc = LabelEncoder()
        lab_enc.fit(labels)
        feature = lab_enc.transform(srs).reshape(len(srs), 1)
        onehot_enc = OneHotEncoder(sparse=False, n_values=len(labels))
        del labels
        categories.append(onehot_enc.fit_transform(feature))

    encoded_cats = np.column_stack(categories)
    del cat_test_df
    del categories

    target = np.log1p(target)
    encoded_df = np.concatenate((encoded_cats, cont_df.values,
                                 target.values), axis=1)
    pd.DataFrame(encoded_df).to_csv("../input/encoded.csv", index=False)


def encode_test(csv_path, n_cat, n_cont):
    test_df = pd.read_csv(csv_path)
    orig_df  = pd.read_csv("../input/train.csv")
    id_df = pd.DataFrame(test_df.iloc[:,0])
    cat_test_df = test_df.iloc[:, 1:n_cat+1]
    cat_df = orig_df.iloc[:, 1:n_cat+1]
    cont_test_df = test_df.iloc[:, n_cat+1:]

    categories = []
    for i in xrange(n_cat):
        srs = cat_test_df.iloc[:,i]
        # Need to know full set of labels for a column. Take from test set.
        labels = list(set(srs.unique().tolist()) |
                      set(cat_df.iloc[:, i].unique().tolist()))
        lab_enc = LabelEncoder()
        lab_enc.fit(labels)
        feature = lab_enc.transform(srs).reshape(len(srs), 1)
        onehot_enc = OneHotEncoder(sparse=False, n_values=len(labels))
        del labels
        categories.append(onehot_enc.fit_transform(feature))

    encoded_cats = np.column_stack(categories)
    del cat_test_df
    del categories

    encoded_df = np.concatenate((id_df.values, encoded_cats, cont_test_df.values), axis=1)
    pd.DataFrame(encoded_df).to_csv("../input/encoded_test.csv", index=False)


encode("../input/train.csv", n_cat=116, n_cont=14)
encode_test("../input/test.csv", n_cat=116, n_cont=14)
print("encode.py finished")
