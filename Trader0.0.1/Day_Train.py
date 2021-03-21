import os

import tensorflow as tf
import numpy as np

import pickle

BATCH_SIZE_PER_REPLICA = 64
EPOCHS = 10
VALIDATION_SPLIT = 0.1

PATH = "../Data/"

SAVE_PATH = "saved_model/day_model"

tf.keras.backend.set_floatx("float64")

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


def save(data, file):
    pickle_out = open(f"{file}.pickle", "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()


def read(file):
    return pickle.load(open(f"{file}.pickle", "rb"))


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


checkpoint_dir = "./day_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir="./day_logs"),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
]

model.fit(train_ds, epochs=EPOCHS, callbacks=callbacks)
model.save(SAVE_PATH)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

eval_loss, eval_acc = model.evaluate(eval_ds)
print(f"Eval loss: {eval_loss}, Eval Accuracy: {eval_acc}")