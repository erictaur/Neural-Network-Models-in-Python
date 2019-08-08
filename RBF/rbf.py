from random import *
import random
import numpy as np
import matplotlib.pyplot as plt


class RBF:
    def __init__(self, x_data, ans_data, c_num, c_acq, som_c_list, update):
        self.data = x_data
        self.ans = ans_data
        self.c_num = c_num
        self.c_acq = c_acq
        self.c_from_som = som_c_list
        self.w_list = None
        self.c_list = None
        self.sigma = None
        self.update = update

    def init_weights(self):
        w = [random.random() for n in range(self.c_num)]
        return w

    def random_c(self):
        c_list = []
        for i in range(self.c_num):
            rand_index = randint(0, len(self.ans)-1)
            c_list.append(self.data[rand_index])
        print('Selected Central Points: \n{0}'.format(c_list))
        return c_list

    def cal_z(self, x):
        z_list = []
        for c in self.c_list:
            temp = np.sum((np.array(x) - np.array(c)) ** 2)
            z = self.phi(temp)
            z_list.append(z)
        return z_list

    @staticmethod
    def cal_y(z, w):
        y = 0
        for i in range(len(w)):
            temp = z[i] * w[i]
            y += temp
        return y

    def cal_sigma(self):
        max_dist = 0
        for i in range(len(self.c_list)):
            for j in range(len(self.c_list)):
                temp = np.sum((np.array(self.c_list[i]) - np.array(self.c_list[j])) ** 2)
                if (i == 0) & (j == 0):
                    max_dist = temp ** 0.5
                elif temp >= max_dist:
                    max_dist = temp ** 0.5
                else:
                    continue
        # print('\nMaximum Distance: {0}\n'.format(max_dist))
        sigma = max_dist / (self.c_num ** 0.5)
        return sigma

    def phi(self, x_min_c_sq):
        value = np.exp(-x_min_c_sq / (2 * (self.sigma ** 2)))
        return value

    @staticmethod
    def cal_e(desired, y):
        return desired - y

    def update_params_sga(self, lr, x, y, des):
        for c, w in zip(self.c_list, self.w_list):
            w += lr * self.cal_e(des, y) * self.phi(np.sum((np.array(x) - np.array(c)) ** 2))
            c += lr * (w * self.cal_e(des, y) * self.phi(np.sum((np.array(x) - np.array(c)) ** 2)) *
                       (np.array(x) - np.array(c)) / (self.sigma ** 2))
            self.sigma += lr * (w * self.cal_e(des, y) * self.phi(np.sum((np.array(x) - np.array(c)) ** 2)) *
                                (np.sum((np.array(x) - np.array(c)) ** 2)) / (self.sigma ** 3))

    @staticmethod
    def binary_y(y):
        bin_y = [0 if y[i] < 0.4 else 1 for i in range(len(y))]
        return bin_y

    def run(self):

        self.w_list = self.init_weights()

        if self.c_acq == 'random':
            self.c_list = self.random_c()

        if self.c_acq == 'som':
            self.c_list = self.c_from_som

        self.sigma = self.cal_sigma()

        epoch = 0
        lr = 0.05

        if self.update == 'lms':
            phi = []
            for x in self.data:
                for c in self.c_list:
                    temp = np.sum((np.array(x) - np.array(c)) ** 2)
                    z = self.phi(temp)
                    phi.append(z)
            phi = np.array(phi).reshape(len(self.data), -1)
            w_star = np.matmul(np.matmul(np.linalg.inv(np.matmul(phi.T, phi)), phi.T), self.ans)
            y = np.matmul(phi, w_star)
            y_list = [0 if y[i] < 0.5 else 1 for i in range(len(y))]
            print('Predicted Y:{0}\n'.format(y_list))
            x_plot = [self.data[m][0] for m in range(len(self.data))]
            y_plot = [self.data[m][1] for m in range(len(self.data))]
            z_val = [y_list[i] for i in range(len(self.ans))]
            z_color = [('#0000FF' if z_val[i] == 0 else ('#00FF00' if z_val[i] == 1 else '#FF0066'))
                       for i in range(len(z_val))]

            for i in range(len(self.data)):
                plt.scatter(x_plot[i], y_plot[i], color=z_color[i], alpha=0.5)

            plt.show()

        if self.update == 'sga':
            while True:
                y_list = []
                E = 0.0
                print('Epoch: {0}'.format(epoch))
                for x, desired, flag in zip(self.data, self.ans, range(len(self.ans))):

                    z_list = self.cal_z(x)
                    y = self.cal_y(z_list, self.w_list)

                    if y < 0.5:
                        y = 0
                    else:
                        y = 1

                    self.update_params_sga(lr, x, y, desired)
                    y_list.append(y)

                    """
                    Calculate Error
                    """
                    e = self.cal_e(desired, y)
                    E += e * e * 0.5

                print('Error: {0}\n'.format(E/len(self.data)))
                if E/len(self.data) < 0.005:
                    print('Predicted Y: {0}\n'.format(y_list))
                    x_plot = [self.data[m][0] for m in range(len(self.data))]
                    y_plot = [self.data[m][1] for m in range(len(self.data))]
                    z_val = [y_list[i] for i in range(len(self.ans))]
                    z_color = [('#0000FF' if z_val[i] == 0 else ('#00FF00' if z_val[i] == 1 else '#FF0066'))
                               for i in range(len(z_val))]

                    for i in range(len(self.data)):
                        plt.scatter(x_plot[i], y_plot[i], color=z_color[i], alpha=0.5)

                    plt.show()
                    break

                if epoch > 200:
                    print('Predicted Y: {0}\n'.format(y_list))
                    x_plot = [self.data[m][0] for m in range(len(self.data))]
                    y_plot = [self.data[m][1] for m in range(len(self.data))]
                    z_val = [y_list[i] for i in range(len(self.ans))]
                    z_color = [('#0000FF' if z_val[i] == 0 else ('#00FF00' if z_val[i] == 1 else '#FF0066'))
                               for i in range(len(z_val))]

                    for i in range(len(self.data)):
                        plt.scatter(x_plot[i], y_plot[i], color=z_color[i], alpha=0.5)

                    plt.show()
                    break

                epoch += 1
                # End of while loop

