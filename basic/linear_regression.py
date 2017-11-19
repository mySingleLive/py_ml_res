

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import style


data_x = np.linspace(-5, 5, 30)
data_y = data_x * 3 - 2
noise = np.random.normal(0, 2, size=30)
data_y = data_y + noise


w = 0.1
b = 0.1


def net(input):
    y = w * input + b
    return y

def loss_func(t, y):
    return ((t - y) ** 2) / 2


epoch = 200
learn_rate = 0.002

data_loss = []

data_w = []
data_b = []


plt.ion()
plt.show()


def train():
    global w, b
    for step in range(epoch):
        pred_y = net(data_x)
        loss = np.mean(((pred_y - data_y) ** 2) / 2)

        # 计算梯度
        # error = diff(loss) = 2 * (1 / 2) * (pred_y - data_y) = pred_y - data_y
        error = pred_y - data_y

        # 推导过程：
        # f(x) = w * x + b
        # diff(f, x) = w
        # diff(f, b) = 1
        # dw = lr * diff(loss, w) = lr * error * diff(f, x) = lr * error * w
        # db = lr * diff(loss, b) = lr * error * diff(f, b) = lr * error * 1
        delta_w = learn_rate * np.sum(error * data_x)
        delta_b = learn_rate * np.sum(error)

        # 更新变量
        w = w - delta_w
        b = b - delta_b
        print('loss = ', loss, 'w = ', w, 'b = ', b)
        plt.cla()
        plt.scatter(data_x, data_y)
        plt.plot(data_x, pred_y, 'r-', lw=2)
        plt.pause(0.3)


train()

print('w = ', w)
print('b = ', b)

# plt.plot([x for x in range(len(data_loss))], data_loss)
# plt.show()

# X = np.linspace(-5, 5, 30)
# Y = forward(X)
#
# plt.plot(X, Y)
# plt.scatter(data_x, data_y, edgecolors='red')

# def painter3D(theta1,theta2,loss):
#     style.use('ggplot')
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111, projection='3d')
#     x,y,z = theta1,theta2,loss
#     ax1.plot_wireframe(x,y,z, rstride=5, cstride=5)



# painter3D(data_w, data_b, data_loss)
# plt.show()

plt.ioff()
