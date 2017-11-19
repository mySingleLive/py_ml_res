
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

EPOCH = 100
LEARN_RATE = 0.02

data_X = np.linspace(-2, 2, 100)
data_Y = 3 * data_X + 2
data_Y = np.random.normal(data_Y, 0.8)

W = tf.Variable(0.0)
B = tf.Variable(0.0)

input_x = tf.placeholder(tf.float32, name="input_x")
true_y = tf.placeholder(tf.float32, name="true_y")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

Y = W * input_x + B

loss = tf.reduce_mean(tf.square(true_y - Y))
optimizer = tf.train.GradientDescentOptimizer(LEARN_RATE)
train = optimizer.minimize(loss)

for step in range(EPOCH):
    sess.run(train, feed_dict={input_x: data_X, true_y: data_Y})
    if step % 10:
        print('step:', step,
              'loss:', sess.run(loss, feed_dict={input_x: data_X, true_y: data_Y}),
              'W:', sess.run(W), 'B:', sess.run(B))


plt.scatter(data_X, data_Y)
predict_Y = sess.run(Y, feed_dict={input_x: data_X})
plt.plot(data_X, predict_Y, color='red')
plt.show()

