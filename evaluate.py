import argparse
import tensorflow as tf

from mnist_conv2d_medium_tutorial import mnist
from mnist_conv2d_medium_tutorial.model import Model

def evaluate():
    with tf.Graph().as_default():
        images, labels = mnist.load_test_data(args.test_data)
        model = Model()

        logits = model.inference(images, keep_prob=1.0)
        accuracy = model.accuracy(logits, labels)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver.restore(sess, args.checkpoint_file_path)

            total_accuracy = sess.run([accuracy])
            print('Test accuracy: {}'.format(total_accuracy))


def main(argv=None):
    evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file_path', type=str,
                        default='checkpoints/model.ckpt-5000',
                        help='path to checkpoint file')
    parser.add_argument('--test_data', type=str,
                        default='data/mnist_test.csv',
                        help='path to test data')

    args = parser.parse_args()
    evaluate(args)