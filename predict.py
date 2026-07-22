import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import tensorflow.compat.v1 as tf
from model import Model
from preprocess_digit import process_image_to_mnist  # import the new function

def predict(image_path, checkpoint_path):
    # 1. Process the image
    input_batch = process_image_to_mnist(image_path)  # shape: (1, 28, 28, 1)

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
            print(f"Predicted digit: {pred[0]}")

if __name__ == "__main__":
    import sys
    import os, os.path

    if len(sys.argv) < 3:
        print(f"Usage:")
        print(f"Usage: python predict.py file <path_to_image>")
        print(f"Usage: python predict.py folder <path_to_folder>")
        sys.exit(1)

    mode = sys.argv[1].lower()
    path = sys.argv[2]
    checkpoint = "checkpoints/model.ckpt"

    if mode == "file":
        if not os.path.isFile(path):
            print(f"Error: {path} is not a valid file")
            sys.exit(1)
        predict(path, checkpoint)

    elif mode == "folder":
        if not os.path.isdir(path):
            print(f"Error: {path} is not a valid folder")
            sys.exit(1)

        image_files = [f for f in os.listdir(path) if f.lower().endswith('.png')]

        if not image_files:
            print(f"No image files found in directory")
            sys.exit(1)

        for filename in image_files:
            full_path = os.path.join(path, filename)
            print("-"*5)
            print(filename)
            print(f"-"*5)
            predict(full_path, checkpoint)

    else:
        print(f"Invalid mode. Use 'file' or 'folder'")
        sys.exit(1)