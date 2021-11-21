from Layers import Layer
import numpy as np

class Activation_Layer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propogation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propogation(self, output_error, learning_rate):
        #print('error : ',output_error)
        #print('input : ',self.input)
        return self.activation_prime(self.input) * (output_error)

