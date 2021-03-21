import os
import pickle
import sys

import tensorflow as tf
import numpy as np
import pandas as pd

from collections import deque
from sklearn import preprocessing

SEQ_LEN = 64
STEP = 1
TARGET = 1.02

BATCH_SIZE_PER_REPLICA = 64
EPOCHS = 10
VALIDATION_SPLIT = 0.1

CLASSIFY = True

PATH = "Data/"
TICKER_LIST = []

SAVE_PATH = "saved_model/day_model"

tf.keras.backend.set_floatx("float64")

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync


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


#### Classify #####


def classify(current, future):
    if float(future) > float(current * TARGET):
        return 1
    else:
        return 0


def classify_df(df):
    df["future"] = df["close"].shift(-1)
    df["target"] = list(map(classify, df["close"], df["future"]))

    df = df.drop(columns=["future"])
    df.fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)
    # df.reset_index(inplace=True, drop=True)
    return df


def preprocess_df(df):
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])

    np.random.shuffle(sequential_data)

    return sequential_data


if CLASSIFY:

    print("Classifying...")

    for filename in os.listdir(PATH):
        TICKER_LIST.append(filename)

    data = []
    for i, TICKER in enumerate(TICKER_LIST):
        progressBar(i, len(TICKER_LIST))
        # print(TICKER)
        df = pd.read_csv(f"{PATH}{TICKER}", index_col="Date")
        df = df.drop(columns=["volume"])
        df.index = pd.to_datetime(df.index)
        df = df.resample("B").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        df = classify_df(df)
        data = data + preprocess_df(df)

    print()
    save(data, "day_data")


#### Train ####


def balance_data(sequential_data):
    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells
    np.random.shuffle(sequential_data)

    return sequential_data


def split_data(sequential_data):
    sequential_data = balance_data(sequential_data)

    x = []
    y = []

    for seq, target in sequential_data:
        x.append(seq)
        y.append(target)

    print(f"Buys: {y.count(1)}, Sells: {y.count(0)}, Buy%: {y.count(1)/len(y)}")

    return np.array(x), y


def create_model():
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(SEQ_LEN, 4)),

            tf.keras.layers.LSTM(128, activation=tf.nn.tanh, recurrent_activation=tf.nn.sigmoid, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.LSTM(128, activation=tf.nn.tanh, recurrent_activation=tf.nn.sigmoid, return_sequences=True),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.LSTM(128, activation=tf.nn.tanh, recurrent_activation=tf.nn.sigmoid),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Dense(32, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(2, activation=tf.nn.softmax)
            ])

        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"])

    return model


def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch < 7:
        return 1e-4
    else:
        return 1e-5


print(f"Number of devices: {strategy.num_replicas_in_sync}")

sequential_data = read("day_data")

split = int(len(sequential_data) * VALIDATION_SPLIT)
data_train = sequential_data[:-split]
data_test = sequential_data[-split:]

train_ds = tf.data.Dataset.from_tensor_slices(split_data(data_train)).cache().batch(BATCH_SIZE)
eval_ds = tf.data.Dataset.from_tensor_slices(split_data(data_test)).batch(BATCH_SIZE)

model = create_model()

class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nLearning rate for epoch {epoch + 1} is {model.optimizer.lr.numpy()}")


checkpoint_dir = "./checkpoints/week_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir="./logs/week_logs"),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
]

model.fit(train_ds, epochs=EPOCHS, callbacks=callbacks)
model.save(SAVE_PATH)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

eval_loss, eval_acc = model.evaluate(eval_ds)
print(f"Eval loss: {eval_loss}, Eval Accuracy: {eval_acc}")