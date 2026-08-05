import tensorflow as tf
import numpy as np
import mnist
import argparse
from model import create_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import tensorflow_addons as tfa

def evaluate(args):
    test_images, test_labels = mnist.load_test_data(args.test_data)

    # Build model and load best weights
    model = create_model(num_labels=10)
    model.build(input_shape=(None, 28, 28, 1))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Load checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path == "checkpoints/model.ckpt" or checkpoint_path.endswith("model.ckpt") and not os.path.exists(checkpoint_path + ".index"):
        latest = tf.train.latest_checkpoint("checkpoints")
        if latest is not None:
            checkpoint_path = latest
            if args.debug: print(f"Using latest checkpoint: {checkpoint_path}")
        else:
            best_path = "checkpoints/model.ckpt_best"
            if os.path.exists(best_path + ".index"):
                checkpoint_path = best_path
                if args.debug: print(f"[DEBUG] Using best checkpoint: {checkpoint_path}")
            else:
                print("[ERROR] No checkpoint found.")
                return
    status = model.load_weights(checkpoint_path)
    status.expect_partial()

    # ---- TTA prediction function ----
    def predict_with_tta(model, images, batch_size, 
                         angles=[-0.1, 0.0, 0.1], 
                         translations=[(-2, 0), (0, 0), (2, 0), (0, -2), (0, 2)]):
        """
        Predict with TTA: original + rotations + translations.
        angles: list of rotation angles in radians.
        translations: list of (dx, dy) translation in pixels.
        """
        num_samples = images.shape[0]
        all_preds = []

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch = images[start:end]

            preds = []
            # Original (no augmentation)
            pred_orig = model.predict(batch, verbose=args.debug)
            preds.append(pred_orig)

            # Rotations
            for angle in angles:
                if angle == 0.0:
                    continue
                rotated = tfa.image.rotate(batch, angle, interpolation='BILINEAR')
                pred = model.predict(rotated, verbose=args.debug)
                preds.append(pred)

            # Translations
            for dx, dy in translations:
                if dx == 0 and dy == 0:
                    continue
                translated = tfa.image.translate(batch, [dx, dy], interpolation='BILINEAR')
                pred = model.predict(translated, verbose=args.debug)
                preds.append(pred)

            # Average all predictions
            avg_pred = np.mean(preds, axis=0)
            all_preds.append(avg_pred)

        return np.vstack(all_preds)

    if args.tta:
        print("Using TTA")
        preds = predict_with_tta(model, test_images, args.batch_size)
    else:
        preds = model.predict(test_images, batch_size=args.batch_size)

    pred_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(test_labels, axis=1)

    # Calculate overall accuracy
    overall_acc = np.mean(pred_classes == true_classes)

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
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model.ckpt',
                        help='path to checkpoint file (or base name, will auto-find latest)')
    parser.add_argument('--test_data', type=str, default='data/emnist_digits_test.csv',
                        help='path to test data')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size for evaluation')
    parser.add_argument('--debug', action="store_true", help='verbose output')
    parser.add_argument('--tta', action="store_false", help='enable test-time augmentation (horizontal flip)')
    args = parser.parse_args()
    evaluate(args)