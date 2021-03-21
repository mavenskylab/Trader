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
STEP = 1
TARGET = 1

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
    if float(future) > float(current * TARGET):
        return 1
    else:
        return 0


def classify_df(df, term):
    df["future"] = df["close"].shift(-1)
    df["target"] = list(map(classify, df["close"], df["future"]))

    df = df.drop(columns=["future"])
    df.fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)
    # df.reset_index(inplace=True, drop=True)
    return df


def preprocess_df(df, seq_len):
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=seq_len)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == seq_len:
            sequential_data.append([np.array(prev_days), i[-1]])

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
day_data = []
week_data = []
month_data = []
for i, TICKER in enumerate(TICKER_LIST):
    progressBar(i, len(TICKER_LIST))
    # print(TICKER)
    df = pd.read_csv(f"{PATH}{TICKER}", index_col="Date")
    df = df.drop(columns=["volume"])
    df.index = pd.to_datetime(df.index)
    day_df = df.resample("B").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    week_df = df.resample("W").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    month_df = df.resample("BMS").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    TARGET = 1
    day_df = classify_df(day_df, 0)
    TARGET = TARGET * 1.05
    week_df = classify_df(week_df, 1)
    TARGET = TARGET * 1.1
    month_df = classify_df(month_df, 2)
    # print(day_df)
    # print(week_df)
    # print(month_df)
    # exit()
    day_data = day_data + preprocess_df(day_df, 64)
    week_data = week_data + preprocess_df(week_df, 32)
    month_data = month_data + preprocess_df(month_df, 8)

print()

save(day_data, "day_data")
save(week_data, "week_data")
save(month_data, "month_data")