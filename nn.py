#
# nn.py: Basic Neural Network implementation stub.  
# You will fill out the stubs below using numpy as much as possible.  
# This class serves as a base for you to build on for the labs.  
#
# Author: Derek Riley, 2020
# Edited By: Stuart Harley
# Inspired by https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
#


import numpy as np


def sigmoid(x):
    """This is the sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """This is the derivative of the sigmoid function."""
    return x * (1 - x)


def relu_function(x):
    """
    An activation function for non-linear unit
    :param x:
    :return:
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """
    The derivative of the ReLU
    :param x:
    :return:
    """
    return 1. * (x > 0)


def tanh_func(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1. - x * x

# # SoftPlus
# def softplus(x):
#     return np.log(1+(np.exp(x)))
#
# # SoftPlus derivative:
# def softplus_deriv(x):
#   return 1/(1+(np.exp(-x))


class NeuralNetwork:
    """Represents a basic fully connected single-layer neural network.  

    Attributes:
        input (2D numpy array): input features, one row for each sample, 
            and one column for each feature
        weights1 (numpy array): connection weights between the input
            and hidden layer
        weights2 (numpy array): connection weights between the hidden
            layer and output neuron
        y (numpy array): expected outputs of the network, one row for each 
            sample, and one column for each output variable
        output (numpy array): stores the current output of the network 
            after a feedforward pass
        learning_rate (float): scales the derivative influence in backprop
    """

    def __init__(self, x, y, num_hidden_neurons=4, lr=1):
        """Setup a Neural Network with a single hidden layer.  This method
        requires two vectors of x and y values as the input and output data.
        """
        self._input = x
        self._weights1 = np.random.rand(self._input.shape[1], num_hidden_neurons)
        self._weights2 = np.random.rand(num_hidden_neurons, 1)
        self._biases1 = np.zeros(num_hidden_neurons)
        self._biases2 = np.zeros(1)
        self._y = y
        self._output = np.zeros(self._y.shape)
        self._learning_rate = lr

    def inference(self, inputs):
        """
        Use the network to make predictions for a given vector of inputs.
        This is the math to support a feedforward pass.  
        """
        self.layer1 = tanh_func(np.dot(inputs, self._weights1) + self._biases1)
        return tanh_func(np.dot(self.layer1, self._weights2) + self._biases2)

    def feedforward(self):
        """
        This is used in the training process to calculate and save the 
        outputs for backpropogation calculations.  
        """
        self._output = self.inference(self._input)

    def backprop(self):
        """
        Update model weights based on the error between the most recent 
        predictions (feedforward) and the training values.  
        """
        # application of the chain rule to find derivatives of the loss function

        derivative2 = np.dot(self.layer1.T, (2 * (self._y - self._output))
                             * tanh_derivative(self._output))

        derivative1 = np.dot(self._input.T, (np.dot(2 * (self._y - self._output) * tanh_derivative(self._output),
                                                    self._weights2.T) * tanh_derivative(self.layer1)))
        # update the weights and biases with the derivatives of the loss function
        self._weights1 += (derivative1 * self._learning_rate)
        self._weights2 += (derivative2 * self._learning_rate)

        biases1_derivative = self._biases1 + (np.sum(
            np.dot(2 * (self._y - self._output) * tanh_derivative(self._output),
                   self._weights2.T) * tanh_derivative(self.layer1),
            axis=0, keepdims=True) * self._learning_rate)

        biases2_derivative = self._biases2 + (np.sum((self._y - self._output) * tanh_derivative(self._output),
                                                    axis=0, keepdims=True) * self._learning_rate)
        # update the biases:
        self._biases1 = biases1_derivative

        self._biases2 = biases2_derivative

    def train(self, epochs=100):
        """This method trains the network for the given number of epochs.
        It doesn't return anything, instead it just updates the state of
        the network variables.
        """
        for epoch in range(epochs):
            self.feedforward()
            self.backprop()

    def loss(self):
        """ Calculate the MSE error for the set of training data."""
        return np.mean(np.square(np.subtract(self._output, self._y)))

    def accuracy_calculator(self):
        """
        A method to calculate the accuracy
        Author: Dr. Riley (he gave us a sample code)
        :return: accuracy
        """
        threshold = 0.8
        tp = np.sum(np.logical_and(self._y == 1, self._output > threshold))
        tn = np.sum(np.logical_and(self._y == 0, self._output <= threshold))
        fp = np.sum(np.logical_and(self._y == 1, self._output <= threshold))
        fn = np.sum(np.logical_and(self._y == 0, self._output > threshold))
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print(self._y)
        print(self._output)
        return accuracy

