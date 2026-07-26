import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import numpy as np
from model import Model
from preprocess_digit import process_image_to_mnist
from opencv import find_contours
import warnings
warnings.filterwarnings('ignore')

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to image file")
    parser.add_argument("--debug", action="store_true", dest='debug', help="debug")
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
        if args.debug: print(f"[DEBUG] Removed file: {f}")

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

    # ---- Step 2: Load model (latest checkpoint) ----
    checkpoint_path = tf.train.latest_checkpoint("checkpoints")
    if checkpoint_path is None:
        best_path = "checkpoints/model.ckpt_best"
        if os.path.exists(best_path + ".index"):
            checkpoint_path = best_path
            if args.debug:
                print(f"[DEBUG] Using best checkpoint: {checkpoint_path}")
        else:
            print("[ERROR] No checkpoint found.")
            return
    else:
        if args.debug:
            print(f"[DEBUG] Using latest checkpoint: {checkpoint_path}")

    model = Model(num_labels=10)
    model.build(input_shape=(None, 28, 28, 1))
    status = model.load_weights(checkpoint_path)
    status.expect_partial()

    # ---- Step 3: Predict each ROI ----
    digits = []
    confidences = []
    conf_thresh = 0.85

    print(f"Original:\t{filename}")
    print("Predicted:\t", end='')

    for roi_path in roi_files:
        # Preprocess ROI → (1, 28, 28, 1)
        img = process_image_to_mnist(roi_path)

        # Predict
        preds = model.predict(img, verbose=0)   # shape (1, 10)
        digit = np.argmax(preds[0])
        conf = np.max(tf.nn.softmax(preds[0]).numpy())

        digits.append(digit)
        confidences.append(conf)

        # Print only if above threshold
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
    print(f"Avg. Accuracy:\t{avg_acc:.2%}")
    print("=" * 30)

if __name__ == "__main__":
    main()