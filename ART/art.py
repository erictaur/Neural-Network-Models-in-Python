import numpy as np
import matplotlib.pyplot as plt


class ART:

    def __init__(self, input_x, w_hat_list, rho, dim, in_num):
        self.x = input_x
        self.w_hat_list = w_hat_list  # Default w_hat_list consists only one w_hat filled with 1
        self.rho = rho
        self.input_dim = dim  # dim * dim = number of pixels of each input
        self.input_num = in_num  # Number of inputs entered

    def flatten_input_and_weights(self):
        # Each element in x and w_hat_list is flattened
        self.x = np.reshape(self.x, (self.input_num, -1)).tolist()
        self.w_hat_list = np.reshape(self.w_hat_list, (1, -1)).tolist()

    @staticmethod
    def cal_w(wh_list):
        w_list = [[(1 / (sum(wh_list[j]) + 0.5)) * wh_list[j][i] for i in range(len(wh_list[0]))] for j in
                  range(len(wh_list))]
        return w_list

    @staticmethod
    def cal_y(x, weight_list):
        y = np.matmul(x, weight_list)
        return y

    @staticmethod
    def cal_v(x, winner_weight_list):
        v = np.matmul(x, winner_weight_list) / (abs(sum(x)))
        return v
    
    def find_winner(self, x, weight_list):
        max_y, max_y_loc = 0, 0
        for i in range(len(weight_list)):
            if i == 0:
                max_y, max_y_loc = self.cal_y(x, weight_list[i]), 0
            # MUST BE "larger or EQUAL to" current maximum in loop
            elif self.cal_y(x, weight_list[i]) >= max_y:
                max_y, max_y_loc = self.cal_y(x, weight_list[i]), i
            else:
                continue
        return max_y, max_y_loc

    def update_w(self, x, loc):
        self.w_hat_list[loc] = [self.w_hat_list[loc] * x for self.w_hat_list[loc], x in zip(self.w_hat_list[loc], x)]

    def create_w(self, x):
        self.w_hat_list.append(x)

    def run_art(self):
        # Convert from visual to computational format
        self.flatten_input_and_weights()
        # Create a list for saving results
        classification_res = [[] for i in range(self.input_num)]

        for x_input, seq in zip(self.x, range(self.input_num + 1)):
            print('Current Input: {0}'.format(x_input))
            w = self.cal_w(self.w_hat_list)
            print('Converted weight list: {0}'.format(w))
            win_y_val, win_y_loc = self.find_winner(x_input, w)
            v = self.cal_v(x_input, self.w_hat_list[win_y_loc])
            print('v value: {0}'.format(v))
            if v >= self.rho:
                self.update_w(x_input, win_y_loc)
                print('updated weight: {0}'.format(self.w_hat_list[win_y_loc]))
                print('Winner node: {0}\n'.format(win_y_loc+1))
                classification_res[win_y_loc].append(seq)
            else:
                self.create_w(x_input)
                print('Created new node: {0}\n'.format(len(self.w_hat_list)))
                classification_res[len(self.w_hat_list) - 1].append(seq)
                continue

        # Reshape to visual format
        # Restricted to 2-D only (see ....reshape())
        print('Complete result after classification: {0}\n'.format(classification_res))
        self.w_hat_list = np.array(self.w_hat_list).reshape(len(self.w_hat_list), self.input_dim, self.input_dim)
        print(self.w_hat_list)

        # Plot the created and adjusted weights of ART
        n_rows, n_cols = self.input_dim, self.input_dim

        for w in self.w_hat_list:
            # Reset the dimensions of image
            image = np.zeros(n_rows * n_cols)
            w = w.ravel()  # Flatten to 1-D array
            for index in range(len(w)):
                image[index] = w[index]  # Fill data to image
            # Set image to suitable dimension for plotting
            image = image.reshape((n_rows, n_cols))
            plt.matshow(image, cmap=plt.cm.bwr)

            plt.gca().set_xticks([x - 0.5 for x in plt.gca().get_xticks()][1:], minor='true')
            plt.gca().set_yticks([y - 0.5 for y in plt.gca().get_yticks()][1:], minor='true')
            plt.grid(which='minor')
            plt.show()

