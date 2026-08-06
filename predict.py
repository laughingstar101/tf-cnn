import tensorflow as tf
import tensorflow_addons as tfa
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import numpy as np
from model import create_model
from preprocess_digit import process_image_to_mnist
from opencv import find_contours
import warnings
warnings.filterwarnings('ignore')

# ---- TTA function (reused from evaluate.py) ----
@tf.function(experimental_relax_shapes=True)
def apply_tta(batch, angles, translations):
    """Apply rotations and translations to a batch and return all augmented versions."""
    augmented = []
    augmented.append(batch)  # original
    for angle in angles:
        if angle != 0.0:
            augmented.append(tfa.image.rotate(batch, angle, interpolation='BILINEAR'))
    for dx, dy in translations:
        if dx != 0 or dy != 0:
            augmented.append(tfa.image.translate(batch, [dx, dy], interpolation='BILINEAR'))
    return tf.stack(augmented, axis=0)  # shape: (num_aug, batch, h, w, c)

def load_models(checkpoint_paths, num_labels=10):
    """Load multiple models from checkpoint paths."""
    models = []
    for ckpt_path in checkpoint_paths:
        model = create_model(num_labels=num_labels)
        model.build(input_shape=(None, 28, 28, 1))
        status = model.load_weights(ckpt_path)
        status.expect_partial()
        models.append(model)
    return models

def predict(model_list, img, tta=True,
            angles=[-0.1, 0.0, 0.1],
            translations=[(-2,0), (0,0), (2,0), (0,-2), (0,2)]):
    """
    Predict with ensemble of models and optional TTA.
    img: (1, 28, 28, 1) numpy array or tensor.
    Returns: (predicted_class, confidence) where confidence is the max softmax probability.
    """
    # Ensure img is a tensor
    if not isinstance(img, tf.Tensor):
        img = tf.convert_to_tensor(img, dtype=tf.float32)

    if tta:
        # Apply augmentations: shape (num_aug, 1, 28, 28, 1)
        aug_imgs = apply_tta(img, angles, translations)
        num_aug = aug_imgs.shape[0]
        flat_imgs = tf.reshape(aug_imgs, (-1, 28, 28, 1))

        # Get predictions from all models on all augmented images
        all_preds = []
        for model in model_list:
            preds = model.predict(flat_imgs, verbose=0)
            preds = tf.reshape(preds, (num_aug, 1, 10))
            avg_pred = tf.reduce_mean(preds, axis=0)
            all_preds.append(avg_pred)

        final_pred = tf.reduce_mean(tf.stack(all_preds, axis=0), axis=0)
    else:
        # Simple ensemble (no TTA)
        all_preds = []
        for model in model_list:
            pred = model.predict(img, verbose=0)
            all_preds.append(pred)
        final_pred = tf.reduce_mean(tf.stack(all_preds, axis=0), axis=0)

    # Convert to numpy and extract class and confidence
    final_pred_np = final_pred.numpy().squeeze()
    pred_class = np.argmax(final_pred_np)
    confidence = np.max(tf.nn.softmax(final_pred_np).numpy())
    return pred_class, confidence

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to image file")
    parser.add_argument("--debug", action="store_true", dest='debug', help="debug")
    parser.add_argument("--no_auto_ensemble", action="store_false", dest='auto_ensemble',
                        help="automatically ensemble all model.ckpt_digits-* checkpoints")
    parser.add_argument("--no_tta", action="store_false", dest='tta',
                        help="enable test-time augmentation (rotation + translation)")
    args = parser.parse_args()

    path = args.image
    if not os.path.isfile(path):
        print(f"[ERROR] {path} is not a valid file")
        return

    # ---- Step 1: Segment digits ----
    find_contours(path, args)

    # Remove old processed debug images
    for f in glob.glob(os.path.join("debug", "processed_*.png")):
        os.remove(f)
        if args.debug:
            print(f"[DEBUG] Removed file: {f}")

    # Read original filename
    info_path = "roi/info.txt"
    if not os.path.isfile(info_path):
        print("[ERROR] info.txt not found.")
        return
    with open(info_path, "r") as f:
        filename = f.readline().strip()

    # Get ROI files
    roi_files = glob.glob("roi/*.png")
    roi_files = [f for f in roi_files if os.path.basename(f).split('.')[0].isdigit()]
    roi_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))

    if not roi_files:
        print("[ERROR] ROI files not found.")
        return

    # ---- Step 2: Load models ----
    if args.auto_ensemble:
        checkpoint_files = glob.glob("checkpoints/model.ckpt_digits-*.index")
        if not checkpoint_files:
            print("[ERROR] No checkpoint files found for auto-ensemble.")
            return
        checkpoints = [f.replace(".index", "") for f in checkpoint_files]
        if args.debug:
            print(f"[DEBUG] Auto-ensemble using {len(checkpoints)} models:")
            for ckpt in checkpoints:
                print(f"  {ckpt}")
        model_list = load_models(checkpoints)
    else:
        checkpoint_path = tf.train.latest_checkpoint("checkpoints")
        if checkpoint_path is None:
            best_path = "checkpoints/model.ckpt_best"
            if os.path.exists(best_path + ".index"):
                checkpoint_path = best_path
            else:
                print("[ERROR] No checkpoint found.")
                return
        if args.debug:
            print(f"[DEBUG] Using checkpoint: {checkpoint_path}")
        model_list = load_models([checkpoint_path])

    if args.debug:
        if args.tta:
            print("Using ensemble + TTA")
        else:
            print("Using ensemble (no TTA)")

    # ---- Step 3: Predict each ROI ----
    digits = []
    confidences = []
    conf_thresh = 0.85

    print(f"Original:\t{filename}")
    print("Predicted:\t", end='')

    for roi_path in roi_files:
        img = process_image_to_mnist(roi_path)

        digit, conf = predict(model_list, img, tta=args.tta)

        digits.append(digit)
        confidences.append(conf)

        if conf >= conf_thresh:
            print(digit, end='')
        else:
            print('?', end='')

    # ---- Step 4: Summary ----
    avg_acc = sum(confidences) / len(confidences)

    print("\n" + "=" * 30)
    print("Per-digit Accuracy Scores")
    print("-" * 30)
    for idx, digit_val in enumerate(digits):
        if confidences[idx] < conf_thresh:
            print(f"{digit_val}: {confidences[idx]:.2%} < {conf_thresh:.2%} (skipped)")
        else:
            print(f"{digit_val}: {confidences[idx]:.2%}")
    print("-" * 30)
    print(f"Avg. Confidence:\t{avg_acc:.2%}")
    print("=" * 30)

if __name__ == "__main__":
    main()