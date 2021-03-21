import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from collections import deque
from sklearn import preprocessing

PATH = "../Test/"
BUY_PATH = "saved_model/buy_model"
SELL_PATH = "saved_model/sell_model"
TICKER_LIST = []

ID = "PKTCYI5JB16QPCWK2HXW"
KEY = "m0jh3SSa2LGFjeeOK8Hf9TjOaCwcDBIhZJMz71NR"

SEQ_LEN = 64
BATCH_SIZE_PER_REPLICA = 512

STARTING_FUNDS = 10000

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

print(f"Number of devices: {strategy.num_replicas_in_sync}")


class TradingAccount():
    def __init__(self):
        self.cash = STARTING_FUNDS
        self.stock = 0
        self.trades = 0


    def trade(self, rate, price):
        tmp = int(self.get_value(price) * rate / (price * 1.005))
        if self.stock < tmp:
            self.buy(tmp - self.stock, price)
        elif self.stock > tmp:
            self.sell(self.stock - tmp, price)
        if tmp != self.stock:
            print(f"{tmp} ≠ {self.stock}")
            exit()
        

    def buy(self, quanity, price):
        if not self.set_cash(self.cash - quanity * price * 1.005):
            print(f"{self.cash - quanity * price * 1.005}: Cash Error!")
            exit()
        if not self.set_stock(self.stock + quanity):
            print(f"{self.stock + quanity}: Stock Error!")
            exit()
        self.trades = self.trades + 1


    def sell(self, quanity, price):
        if not self.set_cash(self.cash + quanity * price * 0.995):
            print(f"{self.cash + quanity * price * 0.995}: Cash Error!")
            exit()
        if not self.set_stock(self.stock - quanity):
            print(f"{self.stock - quanity}: Stock Error!")
            exit()
        self.trades = self.trades + 1

    
    def set_cash(self, cash):
        if cash >= 0:
            self.cash = cash
        return self.cash == cash

    
    def set_stock(self, stock):
        if stock >= 0:
            self.stock = stock
        return self.stock == stock


    def get_value(self, price):
        return self.cash + (self.stock * price)


    def reset(self):
        self.cash = STARTING_FUNDS
        self.stock = 0
        self.trades = 0


def preprocess_df(df):
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
    prev_days = deque(maxlen=SEQ_LEN)

    for date, row in df.iterrows():
        prev_days.append([n for n in row.values[:-2]])
        if len(prev_days) == SEQ_LEN:
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
        replicated_model = tf.keras.models.load_model(path)
        replicated_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy'])

    return replicated_model


account = TradingAccount()

buy_model = load_model(BUY_PATH)
sell_model = load_model(SELL_PATH)

for filename in os.listdir(PATH):
    TICKER_LIST.append(filename)

for i, TICKER in enumerate(TICKER_LIST):
    df = pd.read_csv(f"{PATH}{TICKER}", index_col="Date")
    df = df.drop(columns=["volume"])
    df.index = pd.to_datetime(df.index)

    df.fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)

    df = df.resample("B").agg({"open": "first", "high": "max", "low": "min", "close": "last"})

    d_test, x_test, y_test = preprocess_df(df)

    buy_pred = buy_model.predict(x_test, batch_size=BATCH_SIZE)
    sell_pred = sell_model.predict(x_test, batch_size=BATCH_SIZE)
    
    data = zip(d_test, buy_pred, sell_pred, y_test)

    value = []
    target = []

    account.reset()

    base_stock = int(STARTING_FUNDS/y_test[0][0])

    for date, buy, sell, price in data:
        account.trade(buy[1], price[1])
        print(f"Cash: {account.cash}, Stock: {account.stock}, Value: {account.get_value(price[1])}")

        value.append(account.get_value(price[1]))
        target.append(base_stock * price[1])

    print(f"{TICKER[:-4]}: Cash: {account.cash}, stock: {account.stock}, Value: {account.get_value(y_test[-1][1])}, Trades: {account.trades}")

    plt.title(TICKER[:-4])
    plt.plot(d_test, value, label="value")
    plt.plot(d_test, target, label="target")
    plt.legend()
    plt.show()