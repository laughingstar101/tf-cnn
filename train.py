import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import tensorflow_addons as tfa
import mnist
import argparse
import time
import numpy as np
from model import create_model
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
from sklearn.utils.class_weight import compute_class_weight

def batch_generator(images, labels, batch_size):
    """Yield batches from in-memory arrays without copying the whole dataset."""
    num_samples = len(images)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_idx = indices[start:end]
        yield images[batch_idx], labels[batch_idx]

def train(args):
    # GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Load data
    images, val_images, labels, val_labels, num_classes = mnist.load_train_data(args.train_data)

    model = create_model(num_labels=num_classes)

    learning_rate = 1e-3
    weight_decay = 1e-4
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # ---- Dataset from generator ----
    dataset = tf.data.Dataset.from_generator(
        lambda: batch_generator(images, labels, args.batch_size),
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            tf.TensorShape([None, 28, 28, 1]),
            tf.TensorShape([None, num_classes])
        )
    ).repeat().prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = len(images) // args.batch_size

    # ---- Learning Rate Scheduler (StepLR) ----
    def step_lr_schedule(epoch):
        initial_lr = learning_rate
        drop = 0.5
        epochs_drop = 5
        lr = initial_lr * (drop ** (epoch // epochs_drop))
        return lr

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(step_lr_schedule, verbose=1)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=args.patience,
        min_delta=args.min_delta,
        restore_best_weights=True,
        verbose=1
    )
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(args.summary_dir, time.strftime("%Y%m%d-%H%M%S")),
        histogram_freq=1
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        args.checkpoint + "_best",
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    raw_labels = np.argmax(labels, axis=1)
    class_weights = compute_class_weight('balanced', classes=np.unique(raw_labels), y=raw_labels)
    class_weights = dict(enumerate(class_weights))

    model.fit(
        dataset,
        validation_data=(val_images, val_labels),
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[early_stop, tensorboard, checkpoint, lr_scheduler],
        class_weight=class_weights,
        verbose=1
    )

    model.save_weights(args.checkpoint)
    print("Training finished.")

if __name__ == '__main__':
    epochs = 100
    patience = 10
    min_delta = 0.0001
    batch_size = 512

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model.ckpt')
    parser.add_argument('--train_data', type=str, default='data/emnist_balanced_train.csv')
    parser.add_argument('--summary_dir', type=str, default='graphs')
    parser.add_argument('--patience', type=int, default=patience)
    parser.add_argument('--min_delta', type=float, default=min_delta)
    args = parser.parse_args()
    train(args)