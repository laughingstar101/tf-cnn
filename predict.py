import tensorflow.compat.v1 as tf
from model import Model
from preprocess_digit import process_image_to_mnist  # import the new function

def predict(image_path, checkpoint_path):
    # 1. Process the image perfectly
    input_batch = process_image_to_mnist(image_path)  # shape: (1, 28, 28, 1)

    # 2. Build the graph
    tf.disable_v2_behavior()
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x')
    keep_prob = tf.placeholder(tf.float32, name='dropout_prob')
    model = Model()
    logits = model.inference(x, keep_prob=keep_prob)
    prediction = tf.argmax(logits, axis=1)

    # 3. Restore and predict
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        pred = sess.run(prediction, feed_dict={x: input_batch, keep_prob: 1.0})
        print(f"Predicted digit: {pred[0]}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        predict(sys.argv[1], "checkpoints/model.ckpt-5000-5000")

