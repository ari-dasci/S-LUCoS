import numpy as np
import pandas as pd


def one_hot_encode(x_train, x_test=None):
    """One-hot encode pandas inputs and align test columns to train columns."""
    if isinstance(x_train, np.ndarray):
        return x_train if x_test is None else (x_train, x_test)

    x_train = pd.get_dummies(x_train, drop_first=True).astype("float32")
    if x_test is None:
        return x_train.values

    x_test = pd.get_dummies(x_test, drop_first=True)
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0).astype("float32")
    return x_train.values, x_test.values
