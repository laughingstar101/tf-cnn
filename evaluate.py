import argparse
import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.disable_v2_behavior()

import mnist
from model import Model
import os

def evaluate(args):
    test_images, test_labels = mnist.load_test_data(args.test_data)

    # If no specific checkpoint provided, find the latest one
    checkpoint_path = args.checkpoint_file_path
    if checkpoint_path == "checkpoints/model.ckpt" or checkpoint_path.endswith("model.ckpt") and not os.path.exists(checkpoint_path + ".index"):
        latest = tf.train.latest_checkpoint("checkpoints")
        if latest is not None:
            checkpoint_path = latest
            print(f"Using latest checkpoint: {checkpoint_path}")
        else:
            print("No checkpoint found in checkpoints/ folder.")
            return

    with tf.Graph().as_default():
        x = tf.placeholder(
            shape=[None, mnist.IMAGE_SIZE, mnist.IMAGE_SIZE, 1],
            dtype=tf.float32,
            name='x'
        )
        y = tf.placeholder(
            shape=[None, 10],
            dtype=tf.float32,
            name='y'
        )
        keep_prob = tf.placeholder(tf.float32, name='dropout_prob')

        model = Model()
        logits = model.inference(x, keep_prob=keep_prob)
        
        predictions = tf.argmax(logits, axis=1, name='predictions')
        true_labels = tf.argmax(y, axis=1, name='true_labels')
        accuracy = model.accuracy(logits, y)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, checkpoint_path)

            pred_vals, true_vals, overall_acc = sess.run(
                [predictions, true_labels, accuracy],
                feed_dict={
                    x: test_images,
                    y: test_labels,
                    keep_prob: 1.0
                }
            )

            # ---- Per-Digit Statistics ----
            print("\n" + "="*50)
            print("Per-Digit Classification Breakdown")
            print("="*50)
            print(f"{'Digit':<8} | {'Correct':<10} | {'Wrong':<10} | {'Accuracy'}")
            print("-"*50)

            total_correct = 0
            total_wrong = 0

            for digit in range(10):
                mask = (true_vals == digit)
                total_digit = np.sum(mask)

                if total_digit == 0:
                    correct = 0
                    wrong = 0
                    acc = 0.0
                else:
                    correct = np.sum(pred_vals[mask] == digit)
                    wrong = total_digit - correct
                    acc = correct / total_digit

                total_correct += correct
                total_wrong += wrong

                print(f"{digit:<8} | {correct:<10} | {wrong:<10} | {acc:.2%}")

            print("-"*50)
            print(f"Overall Accuracy (calculated): {total_correct / (total_correct + total_wrong):.4f}")
            print(f"Overall Accuracy (from model):  {overall_acc:.4f}")
            print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint_file_path', 
        type=str,
        default='checkpoints/model.ckpt',
        help='path to checkpoint file (or base name, will auto-find latest)'
    )
    parser.add_argument(
        '--test_data', 
        type=str,
        default='data/emnist_digits_test.csv',
        help='path to test data'
    )

    args = parser.parse_args()
    evaluate(args)