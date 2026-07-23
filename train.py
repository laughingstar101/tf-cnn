import tensorflow.compat.v1 as tf
import argparse
import mnist
from model import Model

NUM_LABELS = 10

def train(args):
    tf.disable_v2_behavior()

    model = Model()
    images, val_images, labels, val_labels = mnist.load_train_data(args.train_data)

    # Training dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(10000).batch(args.batch_size).repeat().prefetch(1)
    iterator = dataset.make_initializable_iterator()
    next_x, next_y = iterator.get_next()

    # Training placeholders
    keep_prob = tf.placeholder(tf.float32, name='dropout_prob')
    global_step = tf.train.get_or_create_global_step()

    logits = model.inference(next_x, keep_prob=keep_prob)
    loss = model.loss(logits=logits, labels=next_y)
    train_acc = model.accuracy(logits, next_y)

    val_x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='val_x')
    val_y = tf.placeholder(tf.float32, shape=[None, 10], name='val_y')
    val_logits = model.inference(val_x, keep_prob=1.0)      # no dropout for validation
    val_accuracy = model.accuracy(val_logits, val_y)

    summary_op = tf.summary.merge_all()
    train_op = model.train(loss, global_step=global_step)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        writer = tf.summary.FileWriter(args.summary_dir, sess.graph)
        sess.run(init)
        sess.run(iterator.initializer)

        for i in range(args.num_iter):
            try:
                if i % 100 == 0:
                    # Training step with summary
                    _, cur_loss, summary = sess.run(
                        [train_op, loss, summary_op],
                        feed_dict={
                            keep_prob: 0.8,
                            val_x: val_images,   # needed because summary includes val summaries
                            val_y: val_labels
                        }
                    )
                    writer.add_summary(summary, i)

                    # Real validation accuracy
                    val_acc = sess.run(
                        val_accuracy,
                        feed_dict={val_x: val_images, val_y: val_labels}
                    )
                    print(f'Iter {i}, loss: {cur_loss:.4f}')
                    print(f'Validation Accuracy: {val_acc:.4f}')
                else:
                    _, cur_loss = sess.run(
                        [train_op, loss],
                        feed_dict={keep_prob: 0.8}
                    )

                if i == args.num_iter - 1:
                    saver.save(sess, args.checkpoint_file_path, global_step)

            except tf.errors.OutOfRangeError:
                break

if __name__ == '__main__':
    num_iters = 15000

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256,
                        help='size of training batches')
    parser.add_argument('--num_iter', type=int, default=num_iters,
                        help='number of training iterations')
    parser.add_argument('--checkpoint_file_path', type=str,
                        default='checkpoints/model.ckpt',
                        help='path to checkpoint file')
    parser.add_argument('--train_data', type=str,
                        default='data/mnist_train.csv',
                        help='path to train and test data')
    parser.add_argument('--summary_dir', type=str,
                        default='graphs',
                        help='path to directory for storing summaries')
    args = parser.parse_args()
    train(args)