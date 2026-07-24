import numpy as np
import pandas as pd

IMAGE_SIZE = 28

def load_train_data(data_path, validation_size=1000):
    # train_data = np.genfromtxt(data_path, delimiter=',', dtype=np.float32)
    train_data = pd.read_csv(data_path, header=None).values.astype(np.float32)
    x_train = train_data[:, 1:] / 255
    y_train = train_data[:, 0]
    y_train = (np.arange(10) == y_train[:, None]).astype(np.float32)

    # Shuffle the data
    indices = np.random.permutation(len(x_train))
    x_train = x_train[indices]
    y_train = y_train[indices]

    # Now split
    x_val = x_train[:validation_size]
    y_val = y_train[:validation_size]
    x_train = x_train[validation_size:]
    y_train = y_train[validation_size:]

    x_train = x_train.reshape(len(x_train), IMAGE_SIZE, IMAGE_SIZE, 1)
    x_val = x_val.reshape(len(x_val), IMAGE_SIZE, IMAGE_SIZE, 1)

    return x_train, x_val, y_train, y_val

def load_test_data(data_path):
    test_data = np.genfromtxt(data_path, delimiter=',', dtype=np.float32)
    x_test = test_data[:, 1:] / 255

    y_test = np.array(test_data[:, 0])
    y_test = (np.arange(10) == y_test[:, None]).astype(np.float32)

    x_test = x_test.reshape(len(x_test), IMAGE_SIZE, IMAGE_SIZE, 1)

    return x_test, y_test