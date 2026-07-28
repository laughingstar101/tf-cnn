import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore

class Model(Model):
    def __init__(self, num_labels=62):
        super(Model, self).__init__()

        self.augment = tf.keras.Sequential([
            layers.RandomTranslation(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)),
            layers.RandomRotation(0.2),
            layers.RandomZoom(height_factor=(-0.2, 0.2)),
            layers.GaussianNoise(stddev=0.1)
        ])

        # Conv layers with ReLU
        self.conv1 = layers.Conv2D(64, (5, 5), padding='same', activation='relu')
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(128, (5, 5), padding='same', activation='relu')
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.conv3 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(512, activation='relu')
        self.dropout = layers.Dropout(0.4)
        self.fc2 = layers.Dense(num_labels) # logits

    def call(self, inputs, training=False):
        x = inputs
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        return self.fc2(x)