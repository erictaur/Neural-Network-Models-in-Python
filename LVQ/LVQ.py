from random import randrange
from math import sqrt
import matplotlib.pyplot as plt


class LVQ:

    def __init__(self, dataset, num_folds, num_codebooks, learning_rate, epochs):
        self.data = dataset
        self.n_codebooks = num_codebooks
        self.n_folds = num_folds
        self.lrate = learning_rate
        self.epochs = epochs

    # Split a dataset into k folds
    def cross_validation_split(self):
        dataset_split = list()
        dataset_copy = list(self.data)
        fold_size = int(len(self.data) / self.n_folds)
        for i in range(self.n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split

    # Calculate accuracy percentage
    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    # Evaluate an algorithm using a cross validation split
    def evaluate_algorithm(self, algorithm, *args):
        # counter for counting which fold the algorithm is currently working on
        count = 0
        folds = self.cross_validation_split()
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            count += 1
            print('{0} fold: \n'.format(count))
            predicted = algorithm(train_set, test_set, count, *args)
            actual = [row[-1] for row in fold]
            accuracy = self.accuracy_metric(actual, predicted)
            scores.append(accuracy)
        return scores

    # calculate the Euclidean distance between two vectors
    def euclidean_distance(self, row1, row2):
        distance = 0.0
        for i in range(len(row1) - 1):
            distance += (row1[i] - row2[i]) ** 2
        return sqrt(distance)

    # Locate the best matching unit
    def get_best_matching_unit(self, codebooks, test_row):
        distances = list()
        for codebook in codebooks:
            dist = self.euclidean_distance(codebook, test_row)
            distances.append((codebook, dist))
        distances.sort(key=lambda tup: tup[1])
        return distances[0][0]

    # Make a prediction with codebook vectors
    def predict(self, codebooks, test_row):
        bmu = self.get_best_matching_unit(codebooks, test_row)
        return bmu[-1]

    # Create a random codebook vector
    def random_codebook(self, train):
        n_records = len(train)
        n_features = len(train[0])
        codebook = [train[randrange(n_records)][i] for i in range(n_features)]
        # adj_codebook = [codebook[i] / max(codebook) for i in range(len(codebook))]
        return codebook

    # Restrict generated codebooks to be in a loosely defined boundary
    # Hypothesis: It is not easy for codebooks to be bmus if they are outside the boundary
    def codebook_check(self, codebook):
        if (codebook[0] > 0) & (codebook[1] > 0):
            return 0
        elif ((codebook[0] > 5) | (codebook[0] < -5)) | ((codebook[1] > 5) | (codebook[1] < -5)):
            return 0
        else:
            return 1

    # Train a set of codebook vectors
    def train_codebooks(self, train, count, n_codebooks, lrate, epochs):
        # codebooks = [random_codebook(train) for i in range(n_codebooks)]
        codebooks = []

        # This flag list is used to check if codebooks are uniformly distributed to all categories
        flag = [0, 0, 0, 0]

        # This for loop is an easy algorithm of distributing codebooks and seeing if they pass the codebook check
        for i in range(10000):
            temp_codebook = self.random_codebook(train)
            if (temp_codebook[2] == 0) & (flag[0] < n_codebooks / 3):
                if self.codebook_check(temp_codebook) == 1:
                        codebooks.append(temp_codebook)
                        flag[0] += 1
                        flag[3] += 1
                else:
                    continue
            elif (temp_codebook[2] == 1) & (flag[1] < n_codebooks / 3):
                if self.codebook_check(temp_codebook) == 1:
                        codebooks.append(temp_codebook)
                        flag[1] += 1
                        flag[3] += 1
                else:
                    continue
            elif (temp_codebook[2] == 2) & (flag[2] < n_codebooks / 3):
                if self.codebook_check(temp_codebook) == 1:
                        codebooks.append(temp_codebook)
                        flag[2] += 1
                        flag[3] += 1
                else:
                    continue
            else:
                continue
            if flag[3] >= n_codebooks:
                init_codebooks = [[codebooks[i][0], codebooks[i][1]] for i in range(len(codebooks))]
                # plt.plot(*zip(*init_codebooks), 'bo')
                # plt.show()
                break

        for epoch in range(epochs):
            # Modify the learning rate after each epoch
            rate = lrate * (1.0 - (epoch / float(epochs)))
            # Run training process through all training data
            for row in train:
                bmu = self.get_best_matching_unit(codebooks, row)
                # Adjust coordinates of the best matching codebook
                for i in range(len(row) - 1):
                    error = row[i] - bmu[i]
                    if bmu[-1] == row[-1]:
                        bmu[i] += rate * error
                    else:
                        bmu[i] -= rate * error
            print('Epoch{0}: {1}\n'.format(epoch, codebooks))

            x_list = [codebooks[m][0] for m in range(len(codebooks))]
            y_list = [codebooks[n][1] for n in range(len(codebooks))]
            color_list = [codebooks[o][2] for o in range(len(codebooks))]
            # Converts '0', '1', '2' to three different colors
            convert_color = [('#0000FF' if color_list[i] == 0 else ('#00FF00' if color_list[i] == 1 else '#FF0066')) for
                             i in range(len(codebooks))]

            # Decides how often this program prints out the progress of the training process
            interval = int(epochs / 5)

            if epoch % interval == 0:
                for q in range(len(codebooks)):
                    plt.scatter(x_list[q], y_list[q], color=convert_color[q], alpha=0.5)
                plt.title('Epoch Number : %d' % epoch)
                plt.show()
        return codebooks

    # LVQ Algorithm V
    def LVQ(self, train, test, count):
        # First, train the network
        codebooks = self.train_codebooks(train, count, self.n_codebooks, self.lrate, self.epochs)
        predictions = list()
        # Then, make predictions with generated test data
        for row in test:
            output = self.predict(codebooks, row)
            predictions.append(output)
        return predictions



