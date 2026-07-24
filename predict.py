import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import tensorflow.compat.v1 as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.disable_v2_behavior()

from model import Model
from preprocess_digit import process_image_to_mnist
from opencv import find_contours

def predict(path, checkpoint_path):
    # 1. Process the image
    input_batch = process_image_to_mnist(path)

    # 2. Build a fresh graph for this prediction
    with tf.Graph().as_default():
        tf.disable_v2_behavior()

        x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x')
        keep_prob = tf.placeholder(tf.float32, name='dropout_prob')
        model = Model()
        logits = model.inference(x, keep_prob=keep_prob)
        prediction = tf.argmax(logits, axis=1)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, checkpoint_path)
            pred = sess.run(prediction, feed_dict={x: input_batch, keep_prob: 1.0})
            return pred[0]

if __name__ == "__main__":
    import sys
    import os
    import argparse
    import glob
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to image file")
    parser.add_argument("--debug", type=int, default=0, help="debug")
    args = parser.parse_args()

    path = args.image

    if not os.path.isfile(path):
        print(f"[ERROR] {path} is not a valid file")
        sys.exit(1)

    find_contours(path, args)

    info_path = "roi/info.txt"
    if not os.path.isfile(info_path):
        print("[ERROR] info.txt not found.")
        sys.exit(1)

    with open(info_path, "r") as f:
        filename = f.readline().strip()
    
    roi_files = glob.glob("roi/*.png")
    roi_files = [f for f in roi_files if os.path.basename(f).split('.')[0].isdigit()]
    roi_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))

    if not roi_files:
        print("[ERROR] ROI files not found.")
        sys.exit(1)

    checkpoint = tf.train.latest_checkpoint("checkpoints")
    if checkpoint is None:
        print("[ERROR] No checkpoint found")
        sys.exit(1)

    print(f"Original:\t{filename}")
    print("Predicted:\t", end='')
    for roi_path in roi_files: # roi_files is a list of roi paths
        roi_name = os.path.basename(roi_path)
        digit = predict(roi_path, checkpoint)
        print(digit, end='')