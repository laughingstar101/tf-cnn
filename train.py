import tensorflow as tf
import tensorflow_addons as tfa
import mnist
import argparse
import os
import time
from model import Model
import warnings
warnings.filterwarnings('ignore')

def train(args):
    # Load data
    images, val_images, labels, val_labels = mnist.load_train_data(args.train_data)

    # Build model
    model = Model(num_labels=10)

    # AdamW optimizer
    optimizer = tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Early stopping callback
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=args.patience,
        min_delta=args.min_delta,
        restore_best_weights=True,
        verbose=1
    )

    # TensorBoard callback
    log_dir = os.path.join(args.summary_dir, time.strftime("%Y%m%d-%H%M%S"))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # ModelCheckpoint callback to save best weights
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        args.checkpoint_file_path + "_best",
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    # Train
    history = model.fit(
        images, labels,
        validation_data=(val_images, val_labels),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=[early_stop, tensorboard, checkpoint],
        verbose=1
    )

    # Save final model (weights only)
    model.save_weights(args.checkpoint_file_path)
    print("Training finished.")

if __name__ == '__main__':
    epochs = 100
    patience = 5
    min_delta = 0.0001

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256,
                        help='size of training batches')
    parser.add_argument('--epochs', type=int, default=epochs,
                        help='number of training iterations')
    parser.add_argument('--checkpoint_file_path', type=str,
                        default='checkpoints/model.ckpt',
                        help='path to checkpoint file')
    parser.add_argument('--train_data', type=str,
                        default='data/emnist_digits_train.csv',
                        help='path to train and test data')
    parser.add_argument('--summary_dir', type=str,
                        default='graphs',
                        help='path to directory for storing summaries')
    parser.add_argument('--patience', type=int, default=patience,
                        help='iterations to wait after last improvement before stopping')
    parser.add_argument('--min_delta', type=float, default=min_delta,
                        help='absolute improvement')
    args = parser.parse_args()
    train(args)