
import numpy as np
import matplotlib.pyplot as plt
# import sklearn.preprocessing as preprocessing


EPOCH = 2000

data_x = np.linspace(-2, 2, 100)
noise = np.random.normal(0, 0.12, 100)
data_y = 0.46 * np.square(data_x) - 0.1 + noise

train_x = data_x[:,None]
train_y = data_y[:,None]
print(train_x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def diff_sigmoid(y):
    return y * (1 - y)

# 1x10
W1 = np.random.uniform(0, 1, [1, 10])
# 1x1
B1 = np.zeros([1, 1]) + 0.1
# 10x1
W2 = np.random.uniform(0, 1, [10, 1])
# 1x1
B2 = np.zeros([1, 1]) + 0.1


def net(inputs):
    global W1, W2, B1, B2
    layer1 = np.matmul(inputs, W1) + B1
    layer1 = sigmoid(layer1)
    out = np.matmul(layer1, W2) + B2
    return layer1, out

plt.ion()
plt.show()


def train(lr):
    global train_x, W1, W2, B1, B2
    for step in range(EPOCH):
        layer1, pred_y = net(train_x)
        loss = np.mean(np.square(pred_y - train_y) / 2)
        error_out = pred_y - train_y

        # 训练输出层权重
        # layer1 = [100x10], layer1.T = [10x100]
        # error_out = [100x1]
        # delta_w2 = layert.T * error_out = [10x1]
        delta_w2 = lr * np.matmul(layer1.T, error_out)
        delta_b2 = lr * np.sum(error_out)

        # 训练隐藏层权重
        # W2 = [10x1], W2.T = [1x10]
        # error_layer1 = error_out * W2.T = [100x10]
        error_layer1 = np.matmul(error_out, W2.T)
        error_sigmoid = error_layer1 * diff_sigmoid(layer1)
        delta_w1 = lr * np.matmul(train_x.T, error_sigmoid)
        delta_b1 = lr * np.sum(error_sigmoid)

        # Update Weights and Biases
        W2 = W2 - delta_w2
        B2 = B2 - delta_b2
        W1 = W1 - delta_w1
        B1 = B1 - delta_b1

        if step % 5 == 0:
            print('loss =', loss)
            plt.cla()
            plt.scatter(train_x, train_y)
            plt.plot(train_x, pred_y, 'r-', lw=5)
            plt.pause(0.001)


train(0.0012)

_, pred_y = net(train_x)


plt.ioff()

plt.scatter(train_x, train_y)
plt.plot(train_x, pred_y, 'r-', lw=5)
plt.show()