import gzip
import numpy as np
import os

def read_idx_images(filename):
    """Read IDX3-ubyte image files (like MNIST/EMNIST)."""
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 784)

def read_idx_labels(filename):
    """Read IDX1-ubyte label files."""
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def convert_emnist_to_csv(prefix='emnist-digits', data_dir='data'):
    train_images_path = os.path.join(data_dir, f'{prefix}-train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(data_dir, f'{prefix}-train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(data_dir, f'{prefix}-test-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(data_dir, f'{prefix}-test-labels-idx1-ubyte.gz')

    print("Loading training images...")
    x_train = read_idx_images(train_images_path)
    print("Loading training labels...")
    y_train = read_idx_labels(train_labels_path)

    print("Loading test images...")
    x_test = read_idx_images(test_images_path)
    print("Loading test labels...")
    y_test = read_idx_labels(test_labels_path)

    train_data = np.column_stack((y_train, x_train))
    test_data = np.column_stack((y_test, x_test))

    train_csv = os.path.join(data_dir, 'emnist_digits_train.csv')
    test_csv = os.path.join(data_dir, 'emnist_digits_test.csv')

    np.savetxt(train_csv, train_data, delimiter=',', fmt='%f')
    np.savetxt(test_csv, test_data, delimiter=',', fmt='%f')

    print(f"Saved {train_csv} with shape {train_data.shape}")
    print(f"Saved {test_csv} with shape {test_data.shape}")

if __name__ == "__main__":
    convert_emnist_to_csv()