import numpy as np
from random import *
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import random


class CPN:
    def __init__(self, data, hidden_num, lr, ans):
        self.x = data
        # (number of hidden neurons at hidden layer)
        self.hid_num = hidden_num
        # (ndarray of weights to hidden layer)
        self.w = None
        # (ndarray of weights to output layer)
        self.pi = None
        self.lr = lr
        self.y = ans

        self.test = None
        self.test_ans = None

    def init_w(self):
        self.w = [np.array([random.random(), random.random()]) for n in range(self.hid_num)]

    def init_pi(self):
        self.pi = [np.array(random.random()) for n in range(self.hid_num)]

    def scale_y(self):

        min_y = 1
        for i in range(len(self.y)):
            if self.y[i] <= min_y:
                min_y = self.y[i]
            else:
                continue

        self.y = np.array([self.y[i]/min_y for i in range(len(self.y))])

    @staticmethod
    def cal_distance(x, w):
        dist = np.sum((np.array(x) - np.array(w)) ** 2)
        return dist

    def find_winner(self, x):
        min_w, min_w_loc = 0, 0
        for idx in range(len(self.w)):
            if idx == 0:
                min_w, min_w_loc = self.cal_distance(x, self.w[idx]), 0
            elif self.cal_distance(x, self.w[idx]) <= min_w:
                min_w, min_w_loc = self.cal_distance(x, self.w[idx]), idx
            else:
                continue
        # print('winner_loc: {0}, min_dist: {1}'.format(min_w_loc, min_w))
        return min_w, min_w_loc

    def update_w(self, w, x):
        w += self.lr * (x - w)

    def update_pi(self, pi, y):
        pi += self.lr * (y - pi)

    def train(self):
        self.init_w()
        self.init_pi()
        self.x = np.array(normalize(np.array(self.x).reshape(1, -1), norm='max')).reshape(len(self.x), -1)
        self.y = np.array(normalize(np.array(self.y).reshape(1, -1), norm='l2')).reshape(len(self.y),)

        epoch = 0
        error_plot = []

        while True:

            y = []
            error = 0

            for x, yd in zip(self.x, self.y):
                min_val, winner_idx = self.find_winner(x)

                self.update_w(self.w[winner_idx], x)
                y.append(self.pi[winner_idx])
                self.update_pi(self.pi[winner_idx], yd)

            epoch += 1

            for n in range(len(y)):
                error += 0.5 * ((np.array(y[n]) - self.y[n]) ** 2)
            error_plot.append(error / len(self.y))

            if epoch % 10 == 0 or (1 <= epoch <= 9):
                print('Epoch:{0}'.format(epoch))
                print(error / len(self.y))

            if error / len(self.y) < 0.0004:
                print('Epoch:{0}'.format(epoch))
                x_plot = [i+1 for i in range(epoch)]
                plt.plot(x_plot, error_plot, 'b-')
                plt.show()
                break
                # print(y)
                # print(self.y)

            if epoch >= 100:
                x_plot = [i+1 for i in range(100)]
                plt.plot(x_plot, error_plot, 'b-')
                plt.show()
                break

    def run(self):
        self.test = self.x[-50:]
        self.test_ans = self.y[-50:]

        test_y = []
        test_error = 0

        # w and Pi not updated in when running the testing set
        for x in self.test:
            min_val, winner_idx = self.find_winner(x)
            test_y.append(self.pi[winner_idx])

        for n in range(len(test_y)):
            test_error += 0.5 * ((np.array(test_y[n]) - self.test_ans[n]) ** 2)

        print('\nTesting Set Error: {0}'.format(test_error / len(self.test_ans)))

        x_plot = [i + 1 for i in range(50)]

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        for i in range(50):
            ax1.scatter(x_plot[i], test_y[i], c='r')
            ax1.scatter(x_plot[i], self.test_ans[i], c='b')
        plt.show()
