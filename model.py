import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore

def create_model(num_labels=62):
    model = tf.keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        # GPU augmentation layers
        layers.RandomTranslation(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)),
        layers.RandomRotation(0.2),
        layers.RandomZoom(height_factor=(-0.2, 0.2)),
        layers.GaussianNoise(stddev=0.1),
        # Convolutional layers
        layers.Conv2D(64, (5, 5), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (5, 5), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_labels)  # logits
    ])
    return model