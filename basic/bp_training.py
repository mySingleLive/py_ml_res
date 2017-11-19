

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sklearn.preprocessing as preprocessing
from mpl_toolkits.mplot3d import axes3d
from matplotlib import style


sample_data = np.array([
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
])

sample_label = np.array([
    [0],
    [1],
    [1],
    [0]
])


W1 = np.random.uniform(0, 2, [3, 4])
W2 = np.random.uniform(0, 2, [4, 1])
epoch = 3000
learn_rate = 0.6
min_loss = 0.05


def true_or_false(x):
    return x > 0.5




def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def diff_sigmoid_with_output(output):
    return output * (1 - output)


def forward(xs):
    global W1, W2
    xs = preprocessing.normalize(xs)
    layer1 = sigmoid(np.dot(xs, W1))
    layer2 = sigmoid(np.dot(layer1, W2))
    return layer1, layer2


def forwardn(xs, ys):
    samples = np.array([xs, ys, np.ones(np.shape(xs))]).T
    return forward(samples)


def train():
    global sample_data, sample_label, W1, W2
    for step in range(epoch):
        layer1, out = forward(sample_data)
        loss = ((sample_label - out) ** 2) / 2
        error = out - sample_label
        out_delta = error * diff_sigmoid_with_output(out)
        layer1_error = np.dot(out_delta, W2.T)
        layer1_delta = layer1_error * diff_sigmoid_with_output(layer1)
        W2 -= learn_rate * np.dot(layer1.T, out_delta)
        W1 -= learn_rate * np.dot(sample_data.T, layer1_delta)
        print('step: ', step)
        print('loss = ', loss)
        if (np.all(loss < min_loss)):
            return

train()

_, Y = forward(sample_data)

SX = sample_data[:, :2]


_, Zn = forward(sample_data)

x = np.linspace(-2, 2, 40)
y = np.linspace(-2, 2, 40)
Xn, Yn = np.meshgrid(x, y)
Zn = []

for i in range(len(Xn)):
    x = Xn[i]
    y = Yn[i]
    _, z = forwardn(Xn[0], Yn[0])
    z = np.reshape(z, [-1])
    Zn.append(z)

Zn = np.array(Zn)


plt.scatter(sample_data[:,0], sample_data[:,1], c=np.reshape(sample_label, [-1]), cmap=cm.hot)

plt.show()

# style.use('ggplot')
# fig = plt.figure()
# ax1 = fig.add_subplot(111, projection='3d')
# ax1.scatter(*SX[0], *sample_label[0])
# ax1.scatter(*SX[1], *sample_label[1], color='red')
# ax1.scatter(*SX[2], *sample_label[2], color='red')
# ax1.scatter(*SX[3], *sample_label[3])
#
#



# ax1.plot_surface(Xn, Yn, Zn, rstride=1, cstride=1, cmap='rainbow')
#
# plt.show()