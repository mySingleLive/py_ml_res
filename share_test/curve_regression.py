
import numpy as np
import matplotlib.pyplot as plt

EPOCH = 2000

data_x = np.linspace(-2, 2, 100)

noise = np.random.normal(0, 0.52, 100)
data_y = 2.3 * np.square(data_x) - 1.2 + noise

# [100] -> [100x1]
train_x = np.reshape(data_x, [100, 1])
train_y = np.reshape(data_y, [100, 1])


W1 = np.random.uniform(0, 2, [1, 10])
B1 = np.zeros([1, 1]) + 0.1
W2 = np.random.uniform(0, 2, [10, 1])
B2 = np.zeros([1, 1]) + 0.1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def diff_sigmoid(y):
    return y * (1 - y)


def net(inputs):
    layer1 = np.matmul(inputs, W1) + B1
    layer1 = sigmoid(layer1)
    out = np.matmul(layer1, W2) + B2
    return layer1, out


plt.ion()
plt.show()


def train(learn_rate):
    global W1, B1, W2, B2
    for step in range(EPOCH):
        layer1, pred_y = net(train_x)
        loss = np.mean(np.square(pred_y - train_y) / 2)

        # error_out = [100x1]
        error_out = pred_y - train_y

        # 输出层
        # layer1 = [100x10], layer1.T = [10x100]
        # W2 = [10x1]
        # delta_W2 = layer1.T dot error_out = [10x1]
        delta_W2 = learn_rate * np.matmul(layer1.T, error_out)
        delta_B2 = learn_rate * np.sum(error_out)

        # 隐藏层
        # W2 = [10x1], W2.T = [1x10]
        # error_out = [100x1]
        # error_layer1 = error_out dot W2 = [100x10]
        error_layer1 = np.matmul(error_out, W2.T)
        # layer1 = [100x10]
        # error_sigmoid = [100x10]
        error_sigmoid = error_layer1 * diff_sigmoid(layer1)
        # train_x = [100x1], W1 = [1x10]
        # delta_W1 = train_x.T dot error_sigmoid = [1x10]
        delta_W1 = learn_rate * np.matmul(train_x.T, error_sigmoid)
        delta_B1 = learn_rate * np.sum(error_sigmoid)

        # Update Weights and Biases
        W2 = W2 - delta_W2
        B2 = B2 - delta_B2
        W1 = W1 - delta_W1
        B1 = B1 - delta_B1

        if step % 10 == 0:
            print('loss =', loss)
            plt.cla()
            plt.scatter(train_x, train_y)
            plt.plot(train_x, pred_y, 'r-', lw=5)
            plt.pause(0.1)


train(0.0012)

plt.ioff()

_, pred_y = net(train_x)
plt.scatter(train_x, train_y)
plt.plot(train_x, pred_y, 'r-', lw=5)
plt.show()



