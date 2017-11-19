
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def linear(inputs, unit_size):
    input_size = inputs.shape[1].value
    Ws = tf.Variable(tf.random_normal([input_size, unit_size]))
    B = tf.Variable(tf.zeros([1, unit_size]) + 0.1)
    return tf.matmul(inputs, Ws) + B


EPOCH = 1000
LEARN_RATE = 0.2

data_X = np.linspace(-2, 2, 100)[:,None]
noise = np.random.normal(0, 0.25, data_X.shape)
data_Y = 1.4 * np.square(data_X) - 0.2 + noise

# N行1列
input_x = tf.placeholder(tf.float32, shape=[None, 1], name='input_x')
# N行1列
true_y = tf.placeholder(tf.float32, shape=[None, 1], name='true_y')

l1 = linear(input_x, 10)
l1 = tf.nn.sigmoid(l1)
net_out = linear(l1, 1)

loss = tf.reduce_mean(tf.square(true_y - net_out))
optimizer = tf.train.GradientDescentOptimizer(LEARN_RATE)
train_step = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.scatter(data_X, data_Y)


for step in range(EPOCH):
    sess.run(train_step, feed_dict={input_x: data_X, true_y: data_Y})
    if step % 20:
        print('step:', step, 'loss:', sess.run(loss, feed_dict={input_x: data_X, true_y: data_Y}))


plt.plot(data_X, sess.run(net_out, feed_dict={input_x: data_X}), color='red')
plt.show()

