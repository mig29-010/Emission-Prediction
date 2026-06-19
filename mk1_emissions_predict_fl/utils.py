import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


FEATURES = [
    "Trace Velocity",
    "EngSpd",
    "MotSpd",
    "GenSpd",
    "EngTrq",
    "MotTrq",
    "GenTrq",
    "Battery Current",
    "Battery power",
    "<BattPwrLoss>"
]

TARGET = "Battery SOC"

SEQ_LEN = 50


def create_sequences(X, y, seq_len):

    X_seq = []
    y_seq = []

    for i in range(len(X) - seq_len):

        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])

    return np.array(X_seq), np.array(y_seq)


def load_dataset(
    csv_path,
    prediction_horizon=3
):

    df = pd.read_csv(csv_path)

    df.columns = df.columns.str.strip()

    df = df.loc[:, ~df.columns.duplicated()]

    df = df[df["MotSpd"] != 0]

    df["Future_SOC"] = df[TARGET].shift(-prediction_horizon)

    df.dropna(inplace=True)

    X = df[FEATURES].values.astype(np.float32)

    y = df["Future_SOC"].values.astype(np.float32)

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    y_mean = y.mean()
    y_std = y.std()

    y = (y - y_mean) / y_std

    X, y = create_sequences(X, y, SEQ_LEN)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        y_mean,
        y_std
    )