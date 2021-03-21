import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

PATH = "../Test/"
BUY_PATH = "saved_model/buy_model"
SELL_PATH = "saved_model/sell_model"
TICKER_LIST = []

SEQ_LEN = 64

BATCH_SIZE_PER_REPLICA = 512

scaler = StandardScaler()
strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


def preprocess_df(df):
    df["test_open"] = df["open"].shift(-1)
    df["test_close"] = df["close"].shift(-1)

    df.fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)

    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']

    df[['volume', 'MACD']] = df[['volume', 'MACD']].pct_change()

    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df[['open', 'high', 'low', 'close']] = scaler.fit_transform(
        df[['open', 'high', 'low', 'close']].to_numpy())

    for col in ['volume', 'MACD']:
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

    df = df[['open', 'high', 'low', 'close',
             'volume', 'MACD', 'test_open', 'test_close']]

    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for date, row in df.iterrows():
        prev_days.append([n for n in row.values[:-2]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append(
                [date, np.array(prev_days), row.values[-2:]])

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
        replicated_model = tf.keras.models.load_model(path)
        replicated_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                 optimizer=tf.keras.optimizers.Adam(),
                                 metrics=['accuracy'])

    return replicated_model


def trade(buy, sell):
    print(f"{buy}, {sell}")
    return np.argmax(buy)
    if buy - sell > 0.75:
        return 1
    elif buy - sell < -0.75:
        return -1
    else:
        return 0


buy_model = load_model(BUY_PATH)
sell_model = load_model(SELL_PATH)

for filename in os.listdir(PATH):
    TICKER_LIST.append(filename)

for i, TICKER in enumerate(TICKER_LIST):
    df = pd.read_csv(f"{PATH}{TICKER}", index_col="Date")
    df.index = pd.to_datetime(df.index)
    vol = df["volume"]
    df = df.resample("B").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})

    df.dropna(inplace=True)

    df["volume"] = np.zeros((len(df),))
    for index, row in vol.items():
        df.loc[index, "volume"] = row

    df["volume"] = df["volume"].replace(0, np.nan)
    df.dropna(inplace=True)

    dates, x_data, y_data = preprocess_df(df)

    buy_pred = buy_model.predict(x_data, batch_size=BATCH_SIZE)
    sell_pred = sell_model.predict(x_data, batch_size=BATCH_SIZE)

    data = zip(dates, buy_pred, sell_pred, y_data)

    value = []
    target = []

    cash = 10000
    stock = 0

    trades = 0

    base_stock = int(cash/y_data[0][0])

    for d, b, s, y in data:
        t = trade(b, s[1])
        if t == 1:
            if stock == 0:
                stock = int(cash/(y[0]))
                cash = cash - (stock * y[0])
                trades = trades + 1
        elif t == -1:
            if stock != 0:
                cash = cash + (stock * y[1])
                stock = 0
                trades = trades + 1

        value.append(cash + stock * y[1])
        target.append(base_stock * y[1])

    print(
        f"Cash: {cash}, Stock: {stock}, Value: {cash + stock * y_data[-1][1]}, Trades: {trades}")

    plt.title(TICKER[:-4])
    plt.plot(dates, value, label="value")
    plt.plot(dates, target, label="target")
    plt.legend()
    plt.show()
