import argparse
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import mnist
from model import Model

def evaluate(args):
    test_images, test_labels = mnist.load_test_data(args.test_data)

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
        accuracy = model.accuracy(logits, y)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, args.checkpoint_file_path)

            total_accuracy = sess.run(
                accuracy, 
                feed_dict={
                    x: test_images,
                    y: test_labels,
                    keep_prob: 1.0
                }
            )
            print(f'Test accuracy: {total_accuracy:.4f}') # ':.4f' likely means format 4 decimal places

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint_file_path', 
        type=str,
        default='checkpoints/model.ckpt-5000-5000',
        help='path to checkpoint file'
    )
    parser.add_argument(
        '--test_data', 
        type=str,
        default='data/mnist_test.csv',
        help='path to test data'
    )

    args = parser.parse_args()
    evaluate(args)