import tensorflow.compat.v1 as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.disable_v2_behavior()
import argparse
import mnist
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model import Model

NUM_LABELS = 10

def train(args):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.summary_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"TensorBoard logs will be saved to: {log_dir}")

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

    best_val_acc = 0.0
    patience_counter = 0
    early_stop = False

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        writer = tf.summary.FileWriter(log_dir, sess.graph)
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

                    if args.early_stopping:
                        if val_acc > best_val_acc + 1e-3:
                            best_val_acc = val_acc
                            patience_counter = 0
                            saver.save(sess, args.checkpoint_file_path + "_best", global_step)
                        else:
                            patience_counter += 1

                        if patience_counter >= args.patience:
                            print(f"Early stopping triggered at iteration {i}")
                            early_stop = True
                            break

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

        if early_stop:
            saver.save(sess, args.checkpoint_file_path, global_step)
            print("Model saved after early stopping.")

if __name__ == '__main__':
    num_iters = 50000
    patience = 10

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256,
                        help='size of training batches')
    parser.add_argument('--num_iter', type=int, default=num_iters,
                        help='number of training iterations')
    parser.add_argument('--checkpoint_file_path', type=str,
                        default='checkpoints/model.ckpt',
                        help='path to checkpoint file')
    parser.add_argument('--train_data', type=str,
                        default='data/emnist_digits_train.csv',
                        help='path to train and test data')
    parser.add_argument('--summary_dir', type=str,
                        default='graphs',
                        help='path to directory for storing summaries')
    parser.add_argument('--early_stopping', action='store_true',
                        help='enable early stopping based on validation accuracy')
    parser.add_argument('--patience', type=int, default=patience,
                        help='iterations to wait after last improvement before stopping')
    args = parser.parse_args()
    train(args)