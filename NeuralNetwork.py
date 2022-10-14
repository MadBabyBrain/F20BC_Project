from audioop import avg
import math
import random
import numpy as np

def activations(act, val):
    acts = {
        1: sigmoid(val),
        2: relu(val),
        3: tangent(val)
    }
    return acts[act]


def activations_derivatives(act, val):
    acts = {
        1: sigmoid_derivative(val),
        2: relu_derivative(val),
        3: tangent_derivative(val)
    }
    return acts[act]


# ================================================================================================== #


def sigmoid(val):
    return 1 / (1 + math.exp(-val))


def relu(val):
    return max(0, val)


def tangent(val):
    return math.tanh(val)


# ================================================================================================== #


def sigmoid_derivative(val):
    return (1 - sigmoid(val)) * sigmoid(val)


def relu_derivative(val):
    out = 0
    if val > 0:
        out = 1
    return out


def tangent_derivative(val):
    return 1 - (math.tanh(val) ** 2)


# ================================================================================================== #


def cost(actual, expected):
    return (actual - expected) ** 2


def cost_array(actual, expected):
    total_cost = 0
    for (a, e) in zip(actual, expected):
        total_cost += cost(a, e)
    return total_cost


def cost_derivative(actual, expected):
    return 2 * (actual - expected)


# ================================================================================================== #


class NeuralNetwork:
    def __init__(self, layers):
        self.layers_size = layers[0]
        self.weights = [[] for x in range(len(self.layers_size) - 1)] # value from input neuron to output neuron for each layer
        self.biases = [[] for x in range(len(self.layers_size) - 1)] # single value added to each node
        self.weighted_outputs = [[] for x in range(len(self.layers_size) - 1)] # sum of previous (layer * weights) + biases
        self.layer_outputs = [[] for x in range(len(self.layers_size) - 1)] # weighted output put through activation function
        self.activations = layers[1] # which activation is being used for each layer
        self.set_arrays()
        pass
    
    def set_arrays(self):
        for layer in range(len(self.layers_size) - 1):
            # set all other arrays 2d-arrays
            self.weights[layer] = [[] for x in range(self.layers_size[layer + 1])] # 
            self.biases[layer] = [0 for x in range(self.layers_size[layer + 1])] # 
            self.weighted_outputs[layer] = [0 for x in range(self.layers_size[layer + 1])] # 
            self.layer_outputs[layer] = [0 for x in range(self.layers_size[layer + 1])] # 
            for node in range(self.layers_size[layer + 1]):
                # set weights array 3d-array
                self.biases[layer][node] = round(random.uniform(-1, 1), 8) # 
                for in_node in range(self.layers_size[layer]):
                    self.weights[layer][node].append(round(random.uniform(-1, 1), 8)) # 
        pass
    
    # a1*w1 + a2*w2 + b
    def calc_weighted_output(self, layer):
        for outnode in range(self.layers_size[layer]):
            total = 0
            for (v, w) in zip(self.layer_outputs[layer - 1], self.weights[layer - 1][outnode]):
                # print(v, w)
                total += v * w
            total += self.biases[layer - 1][outnode]
            self.layer_outputs[layer][outnode] = total
            
        
    # activation_func(^^)
    def calc_node_values(self, layer):
        for outnode in range(self.layers_size[layer]):
            self.layer_outputs[layer][outnode] = activations(self.activations[layer - 1], self.layer_outputs[layer][outnode])
        pass
        
    def train(self, training_data):
        for (input_vec, expected_output) in zip(training_data[0], training_data[1]):
            print(input_vec, expected_output)
            actual = self.eval(input_vec)
            # total_error = cost_array(actual=actual, expected=expected_output)
            self.layer_outputs.insert(0, input_vec)
            

            for outnode in range(self.layers_size[len(self.layers_size) - 1]):
                for end_layer in range(len(self.layers_size) - 1, 0, -1):
                    for current_node in range(self.layers_size[end_layer - 1]):
                        # print(outnode, end_layer - 1, current_node)
                        print(self.weight_recursion([outnode, end_layer - 1, current_node], (actual[outnode], expected_output[outnode])))
                        # self.bias_recursion()

            del self.layer_outputs[0]
        pass

    def weight_recursion(self, l_data, data):
        print(l_data, data, len(self.layers_size) - 1)
        if l_data[1] == len(self.layers_size) - 2:
            val = 1
            val *= cost_derivative(data[0], data[1])
            val *= activations_derivatives(self.activations[l_data[1]], self.weighted_outputs[l_data[1]][l_data[2]])
            return val
        else:
            val = 1
            val *= 1
            val *= 1
            l_data[1] += 1
            return self.weight_recursion(l_data, data) * val
        pass

    def bias_recursion(self, s_layer):
        pass

    def eval(self, input_vec):
        self.layer_outputs.insert(0, input_vec)
        for layer in range(len(self.layers_size) - 1):
            self.calc_weighted_output(layer + 1)
            self.calc_node_values(layer + 1)
        del self.layer_outputs[0]
        return self.layer_outputs[len(self.layer_outputs) - 1]
    
    def print(self):
        print("l_sizes \t", self.layers_size)
        print("weights \t", self.weights)
        print("biases \t\t", self.biases)
        print("weighed_out \t", self.weighted_outputs)
        print("l_outputs \t", self.layer_outputs)
        print("activations \t", self.activations)
    
    
def main():
    n = NeuralNetwork(layers=[[1, 2, 3], [1, 1]])
    n.train([[[0, 0, 0], [0, 0, 1]], [[0, 0, 1], [0, 1, 0]]])
    n.eval([0, 0, 0])
    n.print()
    pass

if __name__ == "__main__":
    main()
