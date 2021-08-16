from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import regularizers
from PIL import Image
from glob import iglob
from tensorflow.keras.utils import to_categorical
import random
from tqdm import tqdm
import numpy as np
import os
import gc
import time

input_1 = Input((768,))
pipeline_1 = BatchNormalization(input_shape=(768,))(input_1)
pipeline_1 = Dense(32, activation="sigmoid",
                   kernel_regularizer=regularizers.l2(0.01),
                   kernel_initializer="he_uniform")(pipeline_1)
pipeline_1 = Dropout(0.9)(pipeline_1)
pipeline_1 = Dense(16, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(pipeline_1)

input_2 = Input((768,))
pipeline_2 = BatchNormalization(input_shape=(768,))(input_2)
pipeline_2 = Dense(32, activation="tanh",
                   kernel_regularizer=regularizers.l2(0.01),
                   kernel_initializer="he_uniform")(pipeline_2)
pipeline_2 = Dropout(0.9)(pipeline_2)
pipeline_2 = Dense(16, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(pipeline_2)

pipe_concat = Concatenate()([pipeline_1, pipeline_2])
merge_1 = Dense(8, activation="sigmoid",
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l2(0.01))(pipe_concat)
output = Dense(2, activation="softmax")(merge_1)

filepath = "Manual Models/mlp_histogram_v17k.hdf5"
model = Model(inputs=[input_1, input_2], outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['categorical_accuracy'])

# model = load_model(filepath)
train_files = [file for file in iglob("Dataset/Images/largetrain_partitions/serialized_histograms/train*")]
test_files = [file for file in iglob("Dataset/Images/largetrain_partitions/serialized_histograms/test*")]

total_len = 0
for test_file in test_files:
    temp = np.load(test_file)
    total_len += temp.shape[0]

val_len = 500000
train = np.zeros((total_len - val_len, 2, 768), dtype='float32')
test = np.zeros((total_len - val_len, 2), dtype="uint8")
val_train = np.zeros((val_len, 2, 768), dtype='float32')
val_test = np.zeros((val_len, 2), dtype="uint8")

_index = 0
for i in range(len(train_files)):
    temp_train = np.load(train_files[i])
    temp_test = np.load(test_files[i])
    temp_train = temp_train.reshape(-1, 2, 768)
    temp_test = temp_test.reshape(-1, 2)
    for j in tqdm(range(temp_train.shape[0]), desc=f"Loading training and test file #{i}, starting at index {_index}"):
        if _index < total_len - val_len:
            train[_index] = temp_train[j]
            test[_index] = temp_test[j]
            _index += 1
        else:
            try:
                _val_index = _index - (total_len - val_len)
                val_train[_val_index] = temp_train[j]
                val_test[_val_index] = temp_test[j]
                _index += 1
            except Exception as e:
                raise Exception(f"{j}, {temp_train.shape[0]}, {_val_index}")

patience = 7
_epoch = 0
_patience_counter = 0
_lowest_val_loss = float('inf')
_batch_size = 32
while True:
    start_time = time.time()

    rand_seed = random.randint(0, 10000)
    np.random.seed(rand_seed)
    np.random.shuffle(train)
    np.random.seed(rand_seed)
    np.random.shuffle(test)

    # _str = ""
    # for i in tqdm(y):
    #     _str += f"{str(i)}\n"
    #
    # file = open("val.txt", 'w')
    # file.write(_str)
    # file.close()
    # break

    history = model.fit([train[:, 0], train[:, 1]], test, batch_size=_batch_size, epochs=1, verbose=1, validation_data=([val_train[:, 0], val_train[:, 1]], val_test))
    # history = model.fit([X_rgb, X_hsv], test, batch_size=_batch_size, epochs=1, validation_split=0.20, verbose=0)

    # print(history.history)
    # print(f"Epoch {_epoch} | "
    #       f"Accuracy: {history.history['acc'][0]} | "
    #       f"Validation Accuracy: {history.history['val_acc'][0]} | "
    #       f"Loss: {history.history['loss'][0]} | "
    #       f"Validation Loss: {history.history['val_loss'][0]} | "
    #       f"Time taken: {time.time() - start_time} seconds")

    print(f"Epoch {_epoch} | "
          f"Accuracy: {history.history['categorical_accuracy'][0]} | "
          f"Validation Accuracy: {history.history['val_categorical_accuracy'][0]} | "
          f"Loss: {history.history['loss'][0]} | "
          f"Validation Loss: {history.history['val_loss'][0]} | "
          f"Time taken: {time.time() - start_time} seconds")

    if history.history['val_loss'][0] < _lowest_val_loss:
        _patience_counter = 0
        _lowest_val_loss = history.history['val_loss']
        model.save(filepath)
    else:
        _batch_size += 2
        _patience_counter += 1
        if _patience_counter == patience:
            break

    _epoch += 1
