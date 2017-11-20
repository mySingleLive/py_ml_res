
import numpy as np
import matplotlib.pyplot as plt


data_x = np.linspace(-5, 5, 40)
noise = np.random.normal(0, 2.1, 40)
data_y = 3 * data_x - 2 + noise

w = np.random.normal(0, 2)
b = 0


def net(inputs):
    return w * inputs + b

plt.ion()

def train(learn_rate):
    global w, b
    for step in range(1000):
        pred_y = net(data_x)
        loss = np.mean(np.square(pred_y - data_y) / 2)

        error = pred_y - data_y
        delta_w = learn_rate * np.dot(error, data_x)
        delta_b = learn_rate * np.dot(error, np.ones(40))

        # Update
        w = w - delta_w
        b = b - delta_b

        print('loss =', loss)
        plt.cla()
        plt.scatter(data_x, data_y)
        plt.plot(data_x, pred_y, 'r-', lw=4)
        plt.pause(0.5)

train(0.002)

plt.ioff()

pred_y = net(data_x)
plt.scatter(data_x, data_y)
plt.plot(data_x, pred_y, 'r-', lw=4)
plt.show()

