# Neural Networks Models in Python
Demonstration of several neural networks with the use of Python 
(Without Third-Party Frameworks)

## ANN (Back Propagation Neural Network)

### Architecture of this package:
This back-propagation (ANN) implementation is designed to handle two inputs and produce a single output.

The width (number of nodes) of the single hidden layer can be configured accordingly.

### Specifications:
The example demo of this ANN is to predict the function output of:
$$
4\cdot(5\sin{x}) + 2y^2
$$
With a range from `1 to 10` for both inputs.

Prediction of different target functions can also be made available provided that the updated training data is feed correctly to the model.

### Requried packages include:
+ Numpy
+ Pandas
+ csv_writer

Normalization used in this specific implementation is the `min-max` method.

### Code Architecture:
Several functions declared in the `Generic_Neural_Network` class are depicted below:
 
```python=3.7
forward(self, X):
//Take in X and progpagate through the network in a feed-forward fashion

backward(self, X, y, o): 
//Calculate deltas and adjust weights.

sigmoid(self, s): 
//Calculate the sigmoid value of parameter s.

sigmoidPrime(self, s): 
//Calculate the derivative of the sigmoid function with parameter s.

train(self, X, y): 
//Define the flow of the ANN network.
```
    
For more info, check the `Word document` in the same directory for details on function declaration and data processing methods.

## SOM (Self organizing map)

## LVQ (Learning vector quantization)

## RBF (Radial Basis Functions)

## ART (Adaptive resonance theory)

## CPN (Counter propagation network)
