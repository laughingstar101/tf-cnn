import tensorflow as tf
import numpy as np
import mnist
import argparse
import glob
from model import create_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import tensorflow_addons as tfa

# ---- Define TTA augmentations as a tf.function to avoid retracing ----
@tf.function(experimental_relax_shapes=True)
def apply_tta(batch, angles, translations):
    """Apply rotations and translations to a batch and return all augmented versions."""
    augmented = []
    augmented.append(batch)
    # Rotations
    for angle in angles:
        if angle != 0.0:
            augmented.append(tfa.image.rotate(batch, angle, interpolation='BILINEAR'))
    # Translations
    for dx, dy in translations:
        if dx != 0 or dy != 0:
            augmented.append(tfa.image.translate(batch, [dx, dy], interpolation='BILINEAR'))
    return tf.stack(augmented, axis=0) # shape: (num_aug, batch, h, w, c)

def evaluate(args):
    test_images, test_labels = mnist.load_test_data(args.test_data)

    if args.auto_ensemble:
        checkpoint_files = glob.glob("checkpoints/model.ckpt_digits-*.index")
        checkpoints = [f.replace(".index", "") for f in checkpoint_files]
        if not checkpoints:
            print("[ERROR] No checkpoint files found for auto-ensemble.")
            return
        if args.debug:
            print(f"Auto-ensemble using {len(checkpoints)} models:")
            for ckpt in checkpoints:
                print(f"  {ckpt}")
    else:
        if args.checkpoint_list:
            checkpoints = args.checkpoint_list.split(',')
        else:
            checkpoints = [args.checkpoint]

    models = []
    for ckpt_path in checkpoints:
        if args.debug:
            print(f"Loading model: {ckpt_path}")
        model = create_model(num_labels=10)
        model.build(input_shape=(None, 28, 28, 1))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        if ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".index"):
            ckpt_path = ckpt_path.replace(".index", "")
        status = model.load_weights(ckpt_path)
        status.expect_partial()
        models.append(model)

    # ---- Predict with Ensemble + TTA ----
    def predict(model_list, images, batch_size,
                                  angles=[-0.1, 0.0, 0.1],
                                  translations=[(-2,0), (0,0), (2,0), (0,-2), (0,2)]):
        """Ensemble predictions with TTA per model."""
        num_samples = images.shape[0]
        all_preds = []

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch = images[start:end]

            # Get all augmented versions (once per batch)
            aug_batch = apply_tta(batch, angles, translations) # (num_aug, batch, 28,28,1)
            num_aug = aug_batch.shape[0]

            # Flatten augmentations
            flat_batch = tf.reshape(aug_batch, (-1, 28, 28, 1))

            # For each model, predict on all augmentations at once
            model_preds = []
            for model in model_list:
                preds = model.predict(flat_batch, verbose=0) # (num_aug * batch, 10)
                # Reshape to (num_aug, batch, 10) and average over augmentations
                preds_reshaped = tf.reshape(preds, (num_aug, -1, 10))
                avg_over_aug = tf.reduce_mean(preds_reshaped, axis=0) # (batch, 10)
                model_preds.append(avg_over_aug)

            avg_over_models = tf.reduce_mean(tf.stack(model_preds, axis=0), axis=0).numpy()
            all_preds.append(avg_over_models)

        return np.vstack(all_preds)

    print(f"Ensemble of {len(models)} models")
    if args.tta:
        if args.debug: print("Using TTA")
        preds = predict(models, test_images, args.batch_size)
    else:
        def simple_ensemble(model_list, images, batch_size):
            all_preds = []
            for start in range(0, len(images), batch_size):
                end = min(start + batch_size, len(images))
                batch = images[start:end]
                batch_preds = []
                for model in model_list:
                    batch_preds.append(model.predict(batch, verbose=0))
                avg_pred = np.mean(batch_preds, axis=0)
                all_preds.append(avg_pred)
            return np.vstack(all_preds)
        preds = simple_ensemble(models, test_images, args.batch_size)

    pred_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(test_labels, axis=1)

    overall_acc = np.mean(pred_classes == true_classes)

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
                        help='path to a single checkpoint file')
    parser.add_argument('--checkpoint_list', type=str, default=None,
                        help='comma-separated list of checkpoint files')
    parser.add_argument('--no_auto_ensemble', action='store_false', dest='auto_ensemble', 
                        help='automatically ensemble all model.ckpt_digits-* checkpoints')
    parser.add_argument('--no_tta', action='store_false', dest='tta',
                        help='enable TTA (rotation + translation) inside the ensemble')
    parser.add_argument('--test_data', type=str, default='data/emnist_digits_test.csv',
                        help='path to test data')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size for evaluation')
    parser.add_argument('--debug', action="store_true", help='verbose output')
    args = parser.parse_args()
    evaluate(args)