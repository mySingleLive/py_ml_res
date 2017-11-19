
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

EPOCH = 800
LEARN_RATE = 0.2

data_X = np.linspace(-2, 2, 200)[:,None]
noise = np.random.normal(0, 0.25, data_X.shape)
data_Y = 1.4 * np.square(data_X) - 0.2 + noise

W1 = tf.Variable(tf.random_normal([1, 10]))
B1 = tf.Variable(tf.zeros([1, 10]) + 0.1)
W2 = tf.Variable(tf.random_normal([10, 1]))
B2 = tf.Variable(tf.zeros([1, 1]) + 0.1)

# N行1列
input_x = tf.placeholder(tf.float32, shape=[None, 1], name='input_x')
# N行1列
true_y = tf.placeholder(tf.float32, shape=[None, 1], name='true_y')

l1 = tf.matmul(input_x, W1) + B1
l1 = tf.nn.sigmoid(l1)
net_out = tf.matmul(l1, W2) + B2

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

plt.plot(data_X, sess.run(net_out, feed_dict={input_x: data_X}), 'r-', lw=5)
plt.show()

