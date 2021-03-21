import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from collections import deque
from sklearn import preprocessing

PATH = "../Test/"
DAY_PATH = "saved_model/buy_model"
WEEK_PATH = "saved_model/week_model"
MONTH_PATH = "saved_model/month_model"
TICKER_LIST = []

BATCH_SIZE_PER_REPLICA = 512

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


def preprocess_df(df, seq_len):
    df["test_open"] = df["open"].shift(-1)
    df["test_close"] = df["close"].shift(-1)

    df.fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)

    for col in df.columns:
        if col != "test_open" and col != "test_close":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=seq_len)

    for date, row in df.iterrows():
        prev_days.append([n for n in row.values[:-2]])
        if len(prev_days) == seq_len:
            sequential_data.append([date, np.array(prev_days), row.values[4:]])

    return split_data(sequential_data)


def split_data(sequential_data):
    d = []
    x = []
    y = []

    for date, seq, test in sequential_data:
        d.append(date)
        x.append(seq)
        y.append(test)

    return d, np.array(x), np.array(y)


def load_model(path):
    with strategy.scope():
        replicated_model = tf.keras.models.load_model(DAY_PATH)
        replicated_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy'])

    return replicated_model


def buy_if(day, week, month):
    return (day + week + month) > 1


def sell_if(day, week, month):
    return (day + week + month) < 2


day_model = load_model(DAY_PATH)
week_model = load_model(WEEK_PATH)
month_model = load_model(MONTH_PATH)

for filename in os.listdir(PATH):
    TICKER_LIST.append(filename)

for i, TICKER in enumerate(TICKER_LIST):
    df = pd.read_csv(f"{PATH}{TICKER}", index_col="Date")
    df = df.drop(columns=["volume"])
    df.index = pd.to_datetime(df.index)

    df.fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)

    day_df = df.resample("B").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    week_df = df.resample("W").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    month_df = df.resample("BMS").agg({"open": "first", "high": "max", "low": "min", "close": "last"})

    d_day, x_day, y_day = preprocess_df(day_df, 64)
    d_week, x_week, y_week = preprocess_df(week_df, 32)
    d_month, x_month, y_month = preprocess_df(month_df, 8)

    day_pred = day_model.predict(x_day, batch_size=BATCH_SIZE)
    week_pred = week_model.predict(x_week, batch_size=BATCH_SIZE)
    month_pred = month_model.predict(x_month, batch_size=BATCH_SIZE)
    
    day_data = zip(d_day, day_pred, y_day)
    week_data = zip(d_week, week_pred, y_week)
    month_data = zip(d_month, month_pred, y_month)

    weekit = iter(week_data)
    monthit = iter(month_data)

    c_weekit = next(weekit)
    c_monthit = next(monthit)

    value = []
    target = []

    cash = 10000
    stock = 0

    trades = 0

    base_stock = int(cash/y_day[0][0])

    for d, x, y in day_data:
        try:
            if d > c_weekit[0]:
                c_weekit = next(weekit)
            if d > c_monthit[0]:
                c_monthit = next(monthit)
        except StopIteration:
            pass
        
        if buy_if(np.argmax(x), np.argmax(c_weekit[1]), np.argmax(c_monthit[1])):
            if stock == 0:
                stock = int(cash/(y[0]))
                cash = cash - (stock * y[0])
                trades = trades + 1
        elif sell_if(np.argmax(x), np.argmax(c_weekit[1]), np.argmax(c_monthit[1])):
            if stock != 0:
                cash = cash + (stock * y[1])
                stock = 0
                trades = trades + 1

        value.append(cash + stock * y[1])
        target.append(base_stock * y[1])

    print(f"Cash: {cash}, Stock: {stock}, Value: {cash + stock * y_day[-1][1]}, Trades: {trades}")

    plt.title(TICKER[:-4])
    plt.plot(d_day, value, label="value")
    plt.plot(d_day, target, label="target")
    plt.legend()
    plt.show()