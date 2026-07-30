import numpy as np
import mnist
import argparse
from model import create_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

def load_mapping(mapping_path='data/emnist-balanced-mapping.txt'):
    mapping = {}
    with open(mapping_path, 'r') as f:
        for line in f:
            idx, ascii_code = line.strip().split()
            mapping[int(idx)] = chr(int(ascii_code))
    return mapping

def evaluate(args):
    test_images, test_labels, num_class = mnist.load_test_data(args.test_data)

    # Build model and load best weights
    model = create_model(num_labels=num_class)
    # model.summary()
    model.build(input_shape=(None, 28, 28, 1))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Load latest checkpoint
    checkpoint_path = args.checkpoint
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

    mapping = load_mapping(args.mapping_file)

    # Per-digit statistics
    print("\n" + "="*50)
    print("Per-Character Classification Breakdown")
    print("="*50)
    print(f"{'Character':<8} | {'Correct':<10} | {'Wrong':<10} | {'Accuracy'}")
    print("-"*50)

    total_correct = 0
    total_wrong = 0
    for cls in range(num_class):
        mask = (true_classes == cls)
        total = np.sum(mask)
        if total == 0:
            correct = wrong = acc = 0
        else:
            correct = np.sum(pred_classes[mask] == cls)
            wrong = total - correct
            acc = correct / total
        total_correct += correct
        total_wrong += wrong
        char = mapping.get(cls, '?')
        print(f"{char:<8} | {correct:<10} | {wrong:<10} | {acc:.2%}")

    print("-"*50)
    print(f"Overall Accuracy: {overall_acc:.2%}")
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model.ckpt', help='path to checkpoint file (or base name, will auto-find latest)')
    parser.add_argument('--test_data', type=str, default='data/emnist_balanced_test.csv', help='path to test data')
    parser.add_argument('--mapping_file', type=str, default='data/emnist-balanced-mapping.txt')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for evaluation (to avoid OOM)')
    parser.add_argument('--debug', action="store_true", dest='debug', help='batch size for evaluation (to avoid OOM)')

    args = parser.parse_args()
    evaluate(args)