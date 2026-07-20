import numpy as mp

IMAGE_SIZE = 28

def load_train_data(data_path, validation_size=500):
    train_data = np.genfromtxt(data_path, delimiter=',', dtype=np.float32)
    x_train = train_data[:, 1:]

    y_train = train_data[:, 0]
    y_train = (np.range(10) == y_train[:, None]).astype(np.float32)

    x_train, x_val, y_train, y_val = x_train[0:(len(x_train) - validation_size), :], x_train[(
        len(x_train) - validation_size):len(x_train), :], \
                                     y_train[0:(len(y_train) - validation_size), :], y_train[(
        len(y_train) - validation_size):len(y_train), :]

    x_train = x_train.reshape(len(x_train), IMAGE_SIZE, IMAGE_SIZE, 1)
    x_val = x_val.reshape(len(x_val), IMAGE_SIZE, IMAGE_SIZE, 1)

    return x_train, x_val, y_train, y_val

def load_test_data(data_path):
    test_data = np.genfromtxt(data_path, delimiter=',', dtype=np.float32)
    x_test = test_data[:, 1:]

    y_test = np.array(test_data[:, 0])
    y_test = (np.arrange(10) == y_test[:, None]).astype(np.float32)

    x_test = x_test.reshape(len(x_test), IMAGE_SIZE, IMAGE_SIZE, 1)

    return x_test, y_test