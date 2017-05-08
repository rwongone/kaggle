from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Lambda, Flatten, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l1, l2
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from subprocess import check_output

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def run(with_test = False):
    """Main entry point."""
    train, test = load_input("../datasets", "train.csv", "test.csv")
    train_rows, train_labels = split_train(train)
    norm_train_rows = norm(train_rows)
    train_images = reshape(norm_train_rows)

    # For CNN
    # model = create_cnn()
    # history = run_model(model, train_images, train_labels)
    # For MLP
    model = create_mlp()
    history = run_model(model, norm_train_rows, train_labels)

    if with_test:
        test_image_rows = test.values.astype('float32')
        test_images = norm(reshape(test_rows))
    else:
        return history


def check_dir(directory):
    print("Found the following files in %s:" % directory)
    print(check_output(["ls", directory]).decode("utf8"))


def load_csv(path):
    print("Loading %s:" % path)
    df = pd.read_csv("%s" % path)
    print("Shape: %s" % (df.shape,))
    return df


def load_input(data_dir, train_filename, test_filename):
    """Load CSVs.

    Returns:
        (pd.DataFrame, pd.DataFrame): train_df and test_df
    """
    check_dir(data_dir)
    train_path = "%s/%s" % (data_dir, train_filename)
    test_path = "%s/%s" % (data_dir, test_filename)

    return load_csv(train_path), load_csv(test_path)


def split_train(train_df):
    """Split training set into data and labels.

    Each row of data is a 784-length vector of brightnesses ranging (0, 255).
    Each label is a single integer ranging 0 to 9, for a digit.

    Returns:
        (np.ndarray, np.ndarray): data, labels.
    """
    return (train_df.ix[:,1:].values.astype('float32'),
            to_categorical(train_df.ix[:,0].values.astype('int32')))


def norm(data):
    """Normalize image brightness by maximum brightness.

    """
    return data / 255


def reshape(data):
    """Convert rows of 784 brightness values into 28x28 images.

    Args:
        data (np.ndarray): image data.

    Returns:
        np.ndarray: reshaped image data.
    """
    return data.reshape(len(data), 28, 28, 1)


def create_cnn(load_hdf5=None):
    """Returns:
        a reference to the compiled model.

    """
    drpout = 0.5
    lmbda = 0
    model = Sequential()
    model.add(Conv2D(32, 3, 3, activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drpout/2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(drpout))
    model.add(Dense(10, activation='softmax'))

    optimizer = RMSprop(lr=0.001)
    loss = "categorical_crossentropy"
    metrics = ["accuracy"]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    if load_hdf5:
        model.load_weights(load_hdf5)

    return model


def create_mlp(load_hdf5=None):
    """Returns:
        a reference to the compiled model.

    """
    drpout = 0.5
    lmbda = 0
    model = Sequential()
    model.add(Dense(800, activation='relu', input_dim=(28*28)))
    model.add(Dropout(drpout))
    model.add(Dense(10, activation='softmax'))

    optimizer = RMSprop(lr=0.001)
    loss = "categorical_crossentropy"
    metrics = ["accuracy"]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    if load_hdf5:
        model.load_weights(load_hdf5)

    return model


def run_model(model, train_images, train_labels):
    """Executes model and returns history."""
    latest = ModelCheckpoint(filepath="../var/weights.latest.hdf5", save_best_only=True)
    return model.fit(train_images,
                     train_labels,
                     validation_split=0.05,
                     nb_epoch=10,
                     batch_size=64,
                     callbacks=[latest]).history


"""Use constant random seed for reproducibility."""
np.random.seed(42)

if __name__ == "__main__":
    run()
