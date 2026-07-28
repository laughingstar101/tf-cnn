import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import glob
import numpy as np
from model import create_model
from preprocess_digit import process_image_to_mnist
from opencv import find_contours
import warnings
warnings.filterwarnings('ignore')

def get_num_classes_from_mapping(mapping_path='data/emnist-byclass-mapping.txt'):
    with open(mapping_path, 'r') as f:
        return sum(1 for _ in f)

def load_mapping(mapping_path='data/emnist-byclass-mapping.txt'):
    mapping = {}
    with open(mapping_path, 'r') as f:
        for line in f:
            idx, ascii_code = line.strip().split()
            mapping[int(idx)] = chr(int(ascii_code))
    return mapping

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

    mapping = load_mapping()

    num_classes = get_num_classes_from_mapping()

    model = create_model(num_labels=num_classes)
    model.build(input_shape=(None, 28, 28, 1))
    status = model.load_weights(checkpoint_path)
    status.expect_partial()

    # ---- Step 3: Predict each ROI ----
    predicted_chars  = []
    confidences = []
    conf_thresh = 0.85

    print(f"Original:\t{filename}")
    print("Predicted:\t", end='')

    for roi_path in roi_files:
        img = process_image_to_mnist(roi_path)
        preds = model.predict(img, verbose=0)
        pred_class = np.argmax(preds[0])
        confidence = np.max(tf.nn.softmax(preds[0]).numpy())
        char = mapping.get(pred_class, '?')
        predicted_chars.append(char)
        confidences.append(confidence)
        print(char, end='')

    print()

    # ---- Step 4: Summary ----
    avg_acc = sum(confidences) / len(confidences) if confidences else 0.0

    print("\n" + "=" * 30)
    print("Per-character Confidence Scores")
    print("-" * 30)
    for idx, char in enumerate(predicted_chars):
        if confidences[idx] < conf_thresh:
            print(f"{char}: {confidences[idx]:.2%} < {conf_thresh:.2%} (skipped)")
        else:
            print(f"{char}: {confidences[idx]:.2%}")
    print("-" * 30)
    print(f"Avg. Confidence:\t{avg_acc:.2%}")
    print("=" * 30)

if __name__ == "__main__":
    main()