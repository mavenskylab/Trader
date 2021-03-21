import os
import pickle
import sys
import time

import tensorflow as tf
import numpy as np
import pandas as pd

from collections import deque
from sklearn import preprocessing

SEQ_LEN = 64
MIN_STEP = 4
MAX_STEP = 32
COMMISSION = 1 + 0

PATH = "../Data/"
TICKER_LIST = []


# pd.set_option("display.max_rows", None)
# pd.set_option("display.max_columns", None)
# pd.set_option("display.width", None)


def save(data, file):
    pickle_out = open(f"{file}.pickle", "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()


def read(file):
    return pickle.load(open(f"{file}.pickle", "rb"))


def progressBar(value, endvalue):
    percent = float(value) / endvalue
    arrow = "-" * int(round(percent * 20) - 1) + ">"
    spaces = " " * (20 - len(arrow))

    sys.stdout.write("\r[{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()


def classify(current, future):
    if float(future) > float(current * COMMISSION):
        return 1
    else:
        return 0


def classify_df(df):
    step = MIN_STEP
    while step <= MAX_STEP:
        df[f"future_{step}"] = df["close"].shift(-step)
        df[f"target_{step}"] = list(map(classify, df["close"], df[f"future_{step}"]))
        df = df.drop(columns=[f"future_{step}"])
        step = step * 2

    df.fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)
    # df.reset_index(inplace=True, drop=True)
    return df


def preprocess_df(df):
    for col in df.columns:
        if "target" not in col:
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_days.append([n for n in i[:-2]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[4]])

    np.random.shuffle(sequential_data)

    return sequential_data


def split_data(sequential_data):
    x = []
    y = []

    for seq, target in sequential_data:
        x.append(seq)
        y.append(target)

    return np.array(x), y


for filename in os.listdir(PATH):
    TICKER_LIST.append(filename)

print("Classifying...")
data = []
for i, TICKER in enumerate(TICKER_LIST):
    progressBar(i, len(TICKER_LIST))
    # print(TICKER)
    df = pd.read_csv(f"{PATH}{TICKER}", index_col="Date")
    df = df.drop(columns=["volume"])
    df.index = pd.to_datetime(df.index)
    df = df.resample("B").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    df = classify_df(df)
    # print(df)
    # exit()
    data = data + preprocess_df(df)


print()

save(data, "data")