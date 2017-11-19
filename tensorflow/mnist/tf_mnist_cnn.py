
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('data', one_hot=True)


def get_accuracy(input_x, true_y, net_out):
    global sess
    pred_y = sess.run(net_out, feed_dict={input_x: input_x})


def linear(inputs, unit_size):
    input_size = inputs.shape[1].value
    Ws = tf.Variable(tf.random_normal([input_size, unit_size]))
    B = tf.Variable(tf.zeros([1, unit_size]) + 0.1)
    return tf.matmul(inputs, Ws) + B


def conv2d(inputs):
    pass

