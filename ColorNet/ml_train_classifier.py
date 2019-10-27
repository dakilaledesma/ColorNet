from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import regularizers
from tensorflow.test import is_gpu_available

from glob import iglob

import random
import time
import gc

print(f"Training using GPU: {is_gpu_available()}")
discriminator_model = Sequential()
discriminator_model.add(Conv2D(16, (3, 3),
                               input_shape=(125, 125, 3),
                               activation="relu"))
discriminator_model.add(Conv2D(16, (5, 5),
                               activation="relu"))
discriminator_model.add(MaxPool2D(pool_size=(2, 2)))
discriminator_model.add(Conv2D(32, (3, 3),
                               activation="relu"))
discriminator_model.add(Conv2D(32, (5, 5),
                               activation="relu"))
discriminator_model.add(MaxPool2D(pool_size=(2, 2)))
discriminator_model.add(BatchNormalization())
discriminator_model.add(Conv2D(64, (3, 3),
                               activation="relu",
                               kernel_initializer="he_uniform"))
discriminator_model.add(Conv2D(64, (5, 5),
                               activation="relu"))
discriminator_model.add(MaxPool2D(pool_size=(2, 2)))
discriminator_model.add(BatchNormalization())
discriminator_model.add(Conv2D(128, (3, 3),
                               activation="relu",
                               kernel_initializer="he_uniform"))
discriminator_model.add(Conv2D(128, (5, 5),
                               activation="relu"))
discriminator_model.add(MaxPool2D(pool_size=(3, 3)))
# discriminator_model.add(Conv2D(2, (1, 1), activation="softmax"))
discriminator_model.add(Flatten())
discriminator_model.add(Dense(256,
                              activation="relu",
                              kernel_regularizer=regularizers.l2(0.001),
                              activity_regularizer=regularizers.l2(0.001)))
discriminator_model.add(Dropout(0.5))
discriminator_model.add(Dense(256,
                              activation="relu",
                              kernel_regularizer=regularizers.l2(0.001),
                              activity_regularizer=regularizers.l2(0.001)))
discriminator_model.add(Dropout(0.5))
discriminator_model.add(Dense(2, activation="softmax"))
discriminator_model.summary()

filepath = "Manual Models/discriminator-v6c.hdf5"
train_files = [file for file in iglob("Dataset/Images/largetrain_partitions/serialized/train*")]
test_files = [file for file in iglob("Dataset/Images/largetrain_partitions/serialized/test*")]

rand_seed = 3
patience = 10

_epoch = 0
_file_int = 0
_patience_counter = 0
_lowest_val_loss = float('inf')
_batch_size = 2
while True:
    # Import numpy within the loop in order to delete it later, allowing garbage collection to collect X and y arrays.
    import numpy as np

    try:
        discriminator_model = load_model(f"{filepath.replace('.hdf5', '')}-tmp.hdf5")
    except OSError:
        try:
            discriminator_model = load_model(filepath)
        except OSError:
            discriminator_model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=['categorical_accuracy'])
    start_time = time.time()

    # X and y are loaded as separate, serialized .npy files due to RAM constraints (64GB).
    X = np.load(train_files[_file_int])
    X /= 255
    y = np.load(test_files[_file_int])

    # Due to X and y being loaded in parts, and validation split being static, random seed must be static during
    # shuffling.
    np.random.seed(rand_seed)
    np.random.shuffle(X)
    np.random.seed(rand_seed)
    np.random.shuffle(y)

    fit_start = time.time()
    history = discriminator_model.fit(X, y, batch_size=_batch_size, epochs=1, validation_split=0.10, verbose=1)

    # print(f"Epoch {_epoch} | "
    #       f"Accuracy: {history.history['categorical_accuracy'][0]} | "
    #       f"Validation Accuracy: {history.history['val_categorical_accuracy'][0]} | "
    #       f"Loss: {history.history['loss'][0]} | "
    #       f"Validation Loss: {history.history['val_loss'][0]} | "
    #       f"Training time taken: {time.time() - fit_start} seconds | "
    #       f"Total time taken: {time.time() - start_time} seconds")

    discriminator_model.save(f"{filepath.replace('.hdf5', '')}-tmp.hdf5")

    # Simple moving batch size inspired by this Google paper: https://arxiv.org/pdf/1711.00489.pdf
    if history.history['val_loss'][0] < _lowest_val_loss:
        _patience_counter = 0
        _lowest_val_loss = history.history['val_loss']
        discriminator_model.save(filepath)
        if _batch_size > 2:
            _batch_size -= 2
    else:
        _patience_counter += 1
        _batch_size += 2
        if _patience_counter == patience:
            break

    # RAM freeing measure, gaining back free space for the next serialized file to be loaded into RAM.
    del history
    del X
    del y
    del discriminator_model
    del np
    gc.collect()

    if _file_int == (len(train_files) - 1):
        _file_int = 0
    else:
        _file_int += 1
    _epoch += 1
