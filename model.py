import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore

def create_model(num_labels=62):
    inputs = tf.keras.Input(shape=(28, 28, 1))

    x = layers.RandomTranslation(height_factor=(-0.05, 0.05),
                                 width_factor=(-0.05, 0.05))(inputs)
    x = layers.RandomRotation(0.05)(x)

    def conv_block(x, filters, num_convs, pool=True):
        for _ in range(num_convs):
            x = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
        if pool:
            x = layers.MaxPooling2D((2, 2))(x)
        return x

    # Block 1: 2×64 → pool
    x = conv_block(x, 64, 2, pool=True)
    # Block 2: 2×128 → pool
    x = conv_block(x, 128, 2, pool=True)
    # Block 3: 3×256 → pool
    x = conv_block(x, 256, 3, pool=True)
    # Block 4: 3×256 → no pool (stays 3x3)
    x = conv_block(x, 384, 4, pool=False)

    x = layers.GlobalAveragePooling2D()(x)

    layer_width = 128
    dropout_rate = 0.2

    orig = x

    aux = layers.Dropout(dropout_rate)(x)

    fc1 = layers.Dense(layer_width, activation='relu')(orig)
    aux = layers.Concatenate()([aux, fc1])

    fc2 = layers.Dense(layer_width, activation='relu')(aux)
    aux = layers.Concatenate()([aux, fc2])

    fc3 = layers.Dense(layer_width, activation='relu')(aux)
    aux = layers.Concatenate()([aux, fc3])

    fc4 = layers.Dense(layer_width, activation='relu')(aux)
    aux = layers.Concatenate()([aux, fc4])

    outputs = layers.Dense(num_labels)(aux)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

    # model = tf.keras.Sequential([
    #     layers.Input(shape=(28, 28, 1)),
    #     # GPU augmentation layers
    #     layers.RandomTranslation(height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05)),
    #     layers.RandomRotation(0.05),
    #     # Convolutional layers
    #     layers.Conv2D(96, (5, 5), padding='same', use_bias=False),
    #     layers.BatchNormalization(),
    #     layers.Activation("relu"),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(192, (5, 5), padding='same', use_bias=False),
    #     layers.BatchNormalization(),
    #     layers.Activation("relu"),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(384, (3, 3), padding='same', use_bias=False),
    #     layers.BatchNormalization(),
    #     layers.Activation("relu"),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(768, (3, 3), padding='same', use_bias=False),
    #     layers.BatchNormalization(),
    #     layers.Activation("relu"),
    #     layers.Flatten(),
    #     layers.Dense(2048, activation='relu'),
    #     layers.Dropout(0.3),
    #     layers.Dense(2048, activation='relu'),
    #     layers.Dropout(0.1),
    #     layers.Dense(2048, activation='relu'),
    #     layers.Dense(1024, activation='relu'),
    #     layers.Dense(num_labels)  # logits
    # ])
    # return model