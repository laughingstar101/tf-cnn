import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore

def create_model(num_labels=10):
    model = tf.keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        # GPU augmentation layers
        layers.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)),
        layers.RandomRotation(0.2),
        layers.RandomZoom(height_factor=(-0.1, 0.1)),
        layers.GaussianNoise(stddev=0.1),
        # Convolutional layers
        layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (5, 5), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_labels),
    ])
    return model