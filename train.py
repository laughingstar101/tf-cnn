import tensorflow.compat.v1 as tf
import argparse
import mnist
from model import Model

NUM_LABELS = 10

def train(args):
    # Disable eager execution to use the 1.x-style session
    tf.disable_v2_behavior()

    model = Model()

    with tf.Graph().as_default():
        images, val_images, labels, val_labels = mnist.load_train_data(args.train_data)

        x = tf.placeholder(shape=[None, mnist.IMAGE_SIZE, mnist.IMAGE_SIZE, 1],
                           dtype=tf.float32, name='x')
        y = tf.placeholder(shape=[None, NUM_LABELS], dtype=tf.float32, name='y')
        keep_prob = tf.placeholder(tf.float32, name='dropout_prob')
        global_step = tf.train.get_or_create_global_step()

        logits = model.inference(x, keep_prob=keep_prob)
        loss = model.loss(logits=logits, labels=y)
        accuracy = model.accuracy(logits, y)

        summary_op = tf.summary.merge_all()
        train_op = model.train(loss, global_step=global_step)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            writer = tf.summary.FileWriter(args.summary_dir, sess.graph)
            sess.run(init)

            for i in range(args.num_iter):
                offset = (i * args.batch_size) % (len(images) - args.batch_size)
                batch_x = images[offset:offset + args.batch_size]
                batch_y = labels[offset:offset + args.batch_size]

                _, cur_loss, summary = sess.run(
                    [train_op, loss, summary_op],
                    feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5}
                )
                writer.add_summary(summary, i)
                print(i, cur_loss)

                if i % 1000 == 0:
                    val_acc = accuracy.eval(feed_dict={
                        x: val_images, y: val_labels, keep_prob: 1.0
                    })
                    print(f'Iter {i} Accuracy: {val_acc}')

                if i == args.num_iter - 1:
                    saver.save(sess, args.checkpoint_file_path, global_step)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128,
                        help='size of training batches')
    parser.add_argument('--num_iter', type=int, default=5000,
                        help='number of training iterations')
    parser.add_argument('--checkpoint_file_path', type=str,
                        default='checkpoints/model.ckpt-5000',
                        help='path to checkpoint file')
    parser.add_argument('--train_data', type=str,
                        default='data/mnist_train.csv',
                        help='path to train and test data')
    parser.add_argument('--summary_dir', type=str,
                        default='graphs',
                        help='path to directory for storing summaries')
    args = parser.parse_args()
    train(args)