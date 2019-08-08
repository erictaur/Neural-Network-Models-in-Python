import numpy as np
import pandas as pd
import csv

"""
Import data generated from data_generator.py
"""
data = pd.read_csv("C:/Users/KHT/Desktop/hw1out.csv", names=["var0", "x", "y", "output"])

X = data.drop(["var0", "output"], axis=1)

"""
Split Data for X1 and X2 and output
"""
X1_nparray = np.asarray(X.iloc[:, 0].tolist())  # Convert dataframe to list then convert list to numpy array

X1_training = X1_nparray[:600]  # First 600 data as training set
X1_test = X1_nparray[600:700]  # last 100 data as testing data

# Same thing for X2
X2_nparray = np.asarray(X.iloc[:, 1].tolist())

X2_training = X2_nparray[:600]
X2_test = X2_nparray[600:700]


X_nparray_training = np.array([row for row in zip(X1_training, X2_training)])
X_nparray_testing = np.array([row for row in zip(X1_test, X2_test)])

y_in = data["output"]

y_nparray = np.reshape(np.asarray(y_in.iloc[:].tolist()), (len(y_in), 1))
y_training = y_nparray[:600]
y_test = y_nparray[600:700]

"""
Scale both the input and output
"""
X_nparray_training = X_nparray_training/np.amax(X_nparray_training, axis=0)  # maximum of X array
y_training = y_training/205  # Scale y by dividing max value of output

X_nparray_testing = X_nparray_testing/np.amax(X_nparray_testing, axis=0)
y_test = y_test/205

"""
Define the class of the BP neural network
"""


class Generic_Neural_Network(object):
    def __init__(self, input_Size, hidden_Size, output_Size, learning_rate):
        # parameters
        self.inputSize = input_Size
        self.outputSize = output_Size
        self.hiddenSize = hidden_Size
        self.learn_rate = learning_rate

        # weights
        # (in_size x hid_size) weight matrix from input to hidden layer
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)

        # (hid_size x out_size) weight matrix from hidden to output layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

    def forward(self, X):
        """
        Forward propagation through network
        """
        # dot product of X (input) and first set of (in_size x hid_size) weights
        self.z = np.matmul(X, self.W1)
        # activate the resulted dot product
        self.z_activated = self.sigmoid(self.z)
        # dot product of hidden layer and second set of (hid_size x out_size) weights
        self.z3 = np.dot(self.z_activated, self.W2)
        o = self.sigmoid(self.z3)  # final activation function
        return o

    def sigmoid(self, s):
        # activation function
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        # derivative of the sigmoid function
        return s * (1 - s)

    def backward(self, X, y, o):
        """
        Back propagation through network
        """
        # Calculate error output and output delta
        self.o_error = y - o  # error in output
        self.o_delta = self.o_error*self.sigmoidPrime(o)  # applying derivative of sigmoid to error
        self.z2_error = self.o_delta.dot(self.W2.T)  # how much hidden our layer weights contribute to output error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z_activated)  # applying derivative of sigmoid to z2 error
        # Adjust Weights
        # adjusting first set (input --> hidden) weights
        self.W1 += self.learn_rate*X.T.dot(self.z2_delta)
        # adjusting second set (hidden --> output) weights
        self.W2 += self.learn_rate*self.z_activated.T.dot(self.o_delta)

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)


result_list = list([])  # Create a list to place metrics

# Instantiate your ANN here with parameters of your choice
hidden_layer_7_NN = Generic_Neural_Network(2, 7, 1, 0.155)

for i in range(10000):  # trains the network 1,0000 times
    # print("\nMSE Loss: \n" + str(np.mean(np.square(y_training - hidden_layer_7_NN.forward(X_nparray_training)))))
    if i == 9999:
        print("BPNN with hidden layer with 7 neurons: ")
        # print("Input: \n" + str(X_nparray_training))
        # print("\nActual Output: \n" + str(y_training) + "\nTransformed Output: \n" + str(205 * y_training))
        print("\nTransformed Output: \n" + str(205 * y_training))
        # print("\nPredicted Output: \n" + str(hidden_layer_7_NN.forward(X_nparray_training)) +
                # "\nTransformed Predicted Output: \n" + str(205 * hidden_layer_7_NN.forward(X_nparray_training)))
        print("\nTransformed Predicted Output: \n" + str(205 * hidden_layer_7_NN.forward(X_nparray_training)))
        print("\nMSE Loss: \n" + str(np.mean(np.square(y_training - hidden_layer_7_NN.forward(X_nparray_training)))))
        print("\n")
        Predicted_output_seven_neuron = [m[0] for m in hidden_layer_7_NN.forward(X_nparray_training)]
        Transformed_predicted_output_seven_neuron = [m[0] for m in 205 * hidden_layer_7_NN.forward(X_nparray_training)]
        # MSE_seven_neuron = [np.mean(np.square(y_training - hidden_layer_7_NN.forward(X_nparray_training)))]
        result_list.append(Predicted_output_seven_neuron)
        result_list.append(Transformed_predicted_output_seven_neuron)
        # result_list.append(MSE_seven_neuron)
        print("\nMSE Loss test: \n" + str(np.mean(np.square(y_test - hidden_layer_7_NN.forward(X_nparray_testing)))))
    hidden_layer_7_NN.train(X_nparray_training, y_training)

# Instantiate your ANN here with parameters of your choice
hidden_layer_5_NN = Generic_Neural_Network(2, 10, 1, 0.1)

for i in range(10000):  # trains the network 1,0000 times
    # print("\nMSE Loss: \n" + str(np.mean(np.square(y_training - hidden_layer_7_NN.forward(X_nparray_training)))))
    if i == 9999:
        print("\nBPNN with hidden layer with 5 neurons: ")
        # print("Input: \n" + str(X_nparray_training))
        # print("\nActual Output: \n" + str(y_training) + "\nTransformed Output: \n" + str(205 * y_training))
        print("\nTransformed Output: \n" + str(205 * y_training))
        # print("\nPredicted Output: \n" + str(hidden_layer_5_NN.forward(X_nparray_training)) +
              # "\nTransformed Predicted Output: \n" + str(205 * hidden_layer_5_NN.forward(X_nparray_training)))
        print("\nTransformed Predicted Output: \n" + str(205 * hidden_layer_5_NN.forward(X_nparray_training)))
        print("\nMSE Loss: \n" + str(
            np.mean(np.square(y_training - hidden_layer_5_NN.forward(X_nparray_training)))))
        print("\n")
        Predicted_output_five_neuron = [m[0] for m in hidden_layer_5_NN.forward(X_nparray_training)]
        Transformed_predicted_output_five_neuron = [m[0] for m in 205 * hidden_layer_5_NN.forward(X_nparray_training)]
        # MSE_five_neuron = [np.mean(np.square(y_training - hidden_layer_5_NN.forward(X_nparray_training)))]
        result_list.append(Predicted_output_five_neuron)
        result_list.append(Transformed_predicted_output_five_neuron)
        # result_list.append(MSE_five_neuron)
    hidden_layer_5_NN.train(X_nparray_training, y_training)


Actual_output_list = [m[0] for m in y_training]
Transformed_actual_output = [m[0] for m in 205 * y_training]

"""
result_list.extend(Actual_output_list + Transformed_actual_output +
                   Predicted_output_five_neuron + Transformed_predicted_output_five_neuron +
                   Predicted_output_seven_neuron + Transformed_predicted_output_seven_neuron)
                   
The extend method does not work for this case
due to the fact that extend() concatenates a list while append() adds a element to a list
"""

result_list.append(Actual_output_list)
result_list.append(Transformed_actual_output)

result_dict = {}

for x in range(len(Actual_output_list)):
    result_dict["row{0}".format(x)] = [n[x] for n in result_list]

with open('C:/Users/KHT/Desktop/hw1result.csv', 'w', newline='') as f:
    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
    for key, values in result_dict.items():
        wr.writerow([key, values[0], values[1], values[2], values[3], values[4], values[5]])
