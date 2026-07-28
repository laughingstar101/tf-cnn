import tensorflow as tf
import tensorflow_addons as tfa
import mnist
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from model import Model
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

def train(args):
    images, val_images, labels, val_labels, num_classes = mnist.load_train_data(args.train_data)

    model = Model(num_labels=num_classes)

    optimizer = tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=args.patience,
        min_delta=args.min_delta,
        restore_best_weights=True,
        verbose=1
    )

    log_dir = os.path.join(args.summary_dir, time.strftime("%Y%m%d-%H%M%S"))
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1,
        profile_batch='2,5'
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        args.checkpoint_file_path + "_best",
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    model.fit(
        images, labels,
        validation_data=(val_images, val_labels),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=[early_stop, tensorboard, checkpoint],
        verbose=1
    )

    model.save_weights(args.checkpoint_file_path)
    print("Training finished.")

if __name__ == '__main__':
    epochs = 100
    patience = 10
    min_delta = 0.0001
    batch_size = 256

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='size of training batches')
    parser.add_argument('--epochs', type=int, default=epochs,
                        help='number of training epochs')
    parser.add_argument('--checkpoint_file_path', type=str,
                        default='checkpoints/model.ckpt',
                        help='path to checkpoint file')
    parser.add_argument('--train_data', type=str,
                        default='data/emnist_byclass_train.csv',
                        help='path to train and test data')
    parser.add_argument('--summary_dir', type=str,
                        default='graphs',
                        help='path to directory for storing summaries')
    parser.add_argument('--patience', type=int, default=patience,
                        help='epochs to wait after last improvement before stopping')
    parser.add_argument('--min_delta', type=float, default=min_delta,
                        help='absolute improvement')
    args = parser.parse_args()
    train(args)