import tensorflow as tf
import numpy as np
import mnist
import argparse
from model import Model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

def evaluate(args):
    test_images, test_labels = mnist.load_test_data(args.test_data)

    # Build model and load best weights
    model = Model(num_labels=10)
    model.build(input_shape=(None, 28, 28, 1))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Load latest checkpoint
    checkpoint_path = args.checkpoint_file_path
    if checkpoint_path == "checkpoints/model.ckpt" or checkpoint_path.endswith("model.ckpt") and not os.path.exists(checkpoint_path + ".index"):
        latest = tf.train.latest_checkpoint("checkpoints")
        if latest is not None:
            checkpoint_path = latest
            if args.debug: print(f"Using latest checkpoint: {checkpoint_path}")
        else:
            # Try best model
            best_path = "checkpoints/model.ckpt_best"
            if os.path.exists(best_path + ".index"):
                checkpoint_path = best_path
                if args.debug: print(f"[DEBUG] Using best checkpoint: {checkpoint_path}")
            else:
                print("[ERROR] No checkpoint found.")
                return
    status = model.load_weights(checkpoint_path)
    status.expect_partial()

    # Evaluate in batches automatically
    _, overall_acc = model.evaluate(test_images, test_labels, batch_size=args.batch_size, verbose=args.debug)

    # Get predictions for per-digit breakdown
    preds = model.predict(test_images, batch_size=args.batch_size)
    pred_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(test_labels, axis=1)

    # Per-digit statistics
    print("\n" + "="*50)
    print("Per-Digit Classification Breakdown")
    print("="*50)
    print(f"{'Digit':<8} | {'Correct':<10} | {'Wrong':<10} | {'Accuracy'}")
    print("-"*50)

    total_correct = 0
    total_wrong = 0
    for digit in range(10):
        mask = (true_classes == digit)
        total_digit = np.sum(mask)
        if total_digit == 0:
            correct = wrong = acc = 0
        else:
            correct = np.sum(pred_classes[mask] == digit)
            wrong = total_digit - correct
            acc = correct / total_digit
        total_correct += correct
        total_wrong += wrong
        print(f"{digit:<8} | {correct:<10} | {wrong:<10} | {acc:.2%}")

    print("-"*50)
    print(f"Overall Accuracy: {overall_acc:.2%}")
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file_path', type=str, default='checkpoints/model.ckpt', help='path to checkpoint file (or base name, will auto-find latest)')
    parser.add_argument('--test_data', type=str, default='data/emnist_digits_test.csv', help='path to test data')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for evaluation (to avoid OOM)')
    parser.add_argument('--debug', action="store_true", dest='debug', help='batch size for evaluation (to avoid OOM)')

    args = parser.parse_args()
    evaluate(args)