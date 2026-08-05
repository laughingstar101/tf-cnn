import tensorflow as tf
import tensorflow_addons as tfa
import mnist
import argparse
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from model import create_model
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

def diagnose_training(history):
    """Analyse training history and print a diagnosis."""
    # Get training and validation metrics
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Find epoch with best validation accuracy
    best_epoch = np.argmax(val_acc)
    best_val_acc = val_acc[best_epoch]
    best_train_acc = train_acc[best_epoch]
    best_val_loss = val_loss[best_epoch]
    best_train_loss = train_loss[best_epoch]

    # Final metrics
    final_train_acc = train_acc[-1]
    final_val_acc = val_acc[-1]
    final_train_loss = train_loss[-1]
    final_val_loss = val_loss[-1]

    # Compute gaps
    acc_gap = final_train_acc - final_val_acc
    loss_gap = final_val_loss - final_train_loss

    print("\n" + "="*50)
    print("TRAINING DIAGNOSIS")
    print("="*50)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best training accuracy: {best_train_acc:.4f}")
    print(f"Final training accuracy: {final_train_acc:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")
    print(f"Accuracy gap (train - val): {acc_gap:.4f}")
    print(f"Loss gap (val - train): {loss_gap:.4f}")

    # Classification
    if acc_gap > 0.05:
        print("\nThe model is *overfitting*.")
    elif acc_gap < -0.02:
        print("\nThe model is *underfitting*.")
    else:
        print("\nThe model is *balanced*.")

    print("="*50 + "\n")

def get_dataset_name(data_path):
    """Extract dataset name from the CSV file path."""
    base = os.path.basename(data_path)
    if 'balanced' in base:
        return 'balanced'
    elif 'digits' in base:
        return 'digits'
    else:
        return 'unknown'

def train(args):
    images, val_images, labels, val_labels = mnist.load_train_data(args.train_data, validation_size=10000)

    model = create_model(num_labels=10)

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

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    log_dir = os.path.join(args.summary_dir, time.strftime("%Y%m%d-%H%M%S"))
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        profile_batch='2,5'
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        args.checkpoint + "_best",
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    history = model.fit(
        images, labels,
        validation_data=(val_images, val_labels),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop, tensorboard, checkpoint, reduce_lr],
        verbose=1
    )

    diagnose_training(history)

    # Save final weights with dataset name and accuracy
    best_val_acc = max(history.history['val_accuracy']) # 0.9018
    # Format: 0.9018 -> 9018
    acc_formatted = int(best_val_acc * 10000)

    dataset_name = get_dataset_name(args.train_data)
    final_checkpoint_name = f"model.ckpt_{dataset_name}-{acc_formatted}"
    final_checkpoint_path = os.path.join('checkpoints', final_checkpoint_name)

    model.save_weights(final_checkpoint_path)
    print(f"Final model saved as: {final_checkpoint_path}")

    model.save_weights(args.checkpoint)
    print("Training finished.")

if __name__ == '__main__':
    epochs = 100
    patience = 10
    min_delta = 0.0001
    batch_size = 512

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='size of training batches')
    parser.add_argument('--epochs', type=int, default=epochs,
                        help='number of training epochs')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/model.ckpt',
                        help='base path for checkpoint files')
    parser.add_argument('--train_data', type=str,
                        default='data/emnist_digits_train.csv',
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