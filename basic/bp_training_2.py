

import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing


X = np.linspace(-10, 10, 50)

RX = np.reshape(X, [-1, 1])
sample_data = RX * [1, 0] + [0, 1]

print('Sample X = ', sample_data)

Y = 2 * X ** 2 + 6 + np.random.uniform(-6, 6, size=50)

sample_label = np.reshape(Y, [-1, 1])

print('Sample Y = ', sample_label)


W1 = np.random.uniform(0, 2, [2, 11])
W2 = np.random.uniform(0, 2, [11, 1])
epoch = 2000
learn_rate = 0.0012
min_loss = 0.5


def relu(x):
    return np.select(x > 0, x, np.zeros(np.shape(x)))


def diff_relu(x):
    return np.select(x > 0, np.ones(np.shape(x)), np.zeros(np.shape(x)))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def diff_sigmoid_with_output(y):
    return y * (1 - y)


def forward(xs):
    global W1, W2
    layer1 = sigmoid(np.dot(xs, W1))
    out = np.dot(layer1, W2)
    return layer1, out


def train():
    global sample_data, sample_label, W1, W2
    sample_data = preprocessing.normalize(sample_data)
    for step in range(epoch):
        layer1, out = forward(sample_data)
        loss = ((sample_label - out) ** 2) / 2
        error = out - sample_label
        out_delta = error
        layer1_error = np.dot(out_delta, W2.T)
        layer1_delta = layer1_error * diff_sigmoid_with_output(layer1)
        w2_delta = np.dot(layer1.T, out_delta)
        w1_delta = np.dot(sample_data.T, layer1_delta)
        W2 += -learn_rate * w2_delta
        W1 += -learn_rate * w1_delta
        print('step: ', step)
        print('loss = ', loss)
        if (np.all(loss < min_loss)):
            return

train()


plt.scatter(X, Y)

X = np.linspace(-10, 10, 100)

RX = np.reshape(X, [-1, 1])
sample_data = RX * [1, 0] + [0, 1]
sample_data = preprocessing.normalize(sample_data)
_, pred = forward(sample_data)
pred = np.reshape(pred, [-1])
print(pred)

plt.plot(X, pred, 'r-', lw=5)
plt.show()
