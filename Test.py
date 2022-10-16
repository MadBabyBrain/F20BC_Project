from lib2to3.pytree import Node
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

class Network:
    
    class Node:
        value = None
        inWeights = None
        bias = None
        weighted_output = None
        activation = None
        position = None
        
        def __init__(self, value, bias, act, pos):
            self.value = value
            self.inWeights = []
            self.bias = bias
            self.weighted_output = 0
            self.activation = act
            self.position = pos
            pass
        
        def calculate_weighted_sum(self):
            total = 0
            for input_w in self.inWeights:
                total += input_w.nodes[0].value * input_w.weight
            total += self.bias
            self.weighted_output = total
            pass
        
        def calculate_output_value(self, act):
            self.value = activations(act, self.weighted_output)
            pass
        
        def print(self):
            print(self.value)
            print(self.bias)
            for w in self.inWeights:
                w.print()
        
    class Weight:
        nodes = None
        weight = None
        
        def __init__(self, nodes, weight):
            self.nodes = nodes
            self.weight = weight
            pass
        
        def print(self):
            # print(self.nodes)
            print(self.weight)
    
    def __init__(self, layers, l_rate):
        self.l_size = layers[0]
        self.activations = layers[1]
        self.nodes = [[] for x in range(len(self.l_size))]
        self.learning_rate = l_rate
        
        self.initNodes()
        
        # for layer in range(len(self.nodes)):
        #     print("l", layer, "nodes", self.nodes[layer])
            # for node in range(len(self.nodes[layer])):
            #     print(layer, node, self.nodes[layer][node])
        
        self.initWeights()
        
        # print()
        # for layer in range(len(self.nodes)):
        #     for node in range(len(self.nodes[layer])):
        #         # print(layer, node, self.nodes[layer][node].inWeights)
        #         for weight in range(len(self.nodes[layer][node].inWeights)):
        #             print("l", layer, "out", node, "w", weight, "in", self.nodes[layer][node].inWeights[weight].nodes)
        pass
    
    def initNodes(self):
        for layer in range(len(self.l_size)):
            self.nodes[layer] = [None for x in range(self.l_size[layer])]
            for outNode in range(self.l_size[layer]):
                # print(layer, inNode, outNode)
                self.nodes[layer][outNode] = self.Node(0, random.uniform(-1, 1), self.activations[layer - 1], (layer, outNode))
                pass
            pass
        pass
    
    def initWeights(self):
        for layer in range(1, len(self.l_size)):
            for inNode in range(0, self.l_size[layer - 1]):
                for outNode in range(0, self.l_size[layer]):
                    # print(layer, inNode, outNode)
                    # print("in", self.nodes[layer - 1], "out", self.nodes[layer])
                    iNode = self.nodes[layer - 1][inNode]
                    oNode = self.nodes[layer][outNode]
                    w = self.Weight((iNode, oNode), random.uniform(-1, 1))
                    self.nodes[layer][outNode].inWeights.append(w)
                    # for w in self.nodes[layer][outNode].inWeights:
                    #     print(w.weight)
                    pass
                pass
            pass
        pass
    
    def train(self, training_data):
        for (input_vec, expected_output) in zip(training_data[0], training_data[1]):
            actual = self.eval(input_vec)
            # print(cost_array(actual, expected_output))
            # while(cost_array(actual, expected_output) > 0.1):
            for depth in range(len(self.l_size) - 1):
                for (outNode, num) in zip(self.nodes[len(self.nodes) - 1], range(len(self.nodes[len(self.nodes) - 1]))):
                    self.weight_recursion(outNode, outNode, depth, 0, expected_output[num], 1)
                    self.bias_recursion(outNode, outNode, depth, 0, expected_output[num], 1)
                    pass
                pass
            pass
            # pass
        pass
    
    def weight_recursion(self, s_node, c_node, depth, current, expected, value):
        for weight in c_node.inWeights:
            if current != depth:
                val = 1
                val *= weight.weight
                val *= weight.nodes[0].value
                self.weight_recursion(s_node, weight.nodes[0], depth, current + 1, expected, (value * val))
            else:
                val = 1
                val *= cost_derivative(s_node.value, expected)
                val *= activations_derivatives(c_node.activation, c_node.value)
                val *= c_node.value
                weight.weight -= self.learning_rate * (value * val)
                
    def bias_recursion(self, s_node, c_node, depth, current, expected, value):
        for weight in c_node.inWeights:
            if current < depth:
                val = 1
                val *= weight.weight
                val *= weight.nodes[0].value
                self.weight_recursion(s_node, weight.nodes[0], depth, current + 1, expected, (value * val))
            else:
                val = 1
                val *= cost_derivative(s_node.value, expected)
                val *= activations_derivatives(c_node.activation, c_node.value)
                val *= 1
                
                self.nodes[c_node.position[0]][c_node.position[1]].bias -= self.learning_rate * (value * val)

    
    def eval(self, input_vec):
        for val in range(len(input_vec)):
            self.nodes[0][val].value = input_vec[val]
            
       
        for layer in range(1, len(self.l_size)):
            for node in range(self.l_size[layer]):
                self.nodes[layer][node].calculate_weighted_sum()
                self.nodes[layer][node].calculate_output_value(self.activations[layer - 1])
                pass
            pass
        
        # for layer in range(len(self.nodes)):
        #     for node in range(len(self.nodes[layer])):
        #         print(layer, node, self.nodes[layer][node].value)
            
        output = []    
        for node in range(len(self.nodes[len(self.nodes) - 1])):
            output.append(self.nodes[len(self.nodes) - 1][node].value)
        return output
    

def main():
    n = Network(layers=[[2, 2, 4], [1, 1]], l_rate=0.01)
    # output.index(max(output))
    o1 = n.eval([1, 0])
    o2 = n.eval([0, 1])
    
    print("e:", 2, "a:", o1.index(max(o1)) + 1, "\te:", 4, "a:", o2.index(max(o2)) + 1)
    
    for x in range(100000):
        n.train([[[1,0], [0,1]], [[0,1,0,0], [0,0,0,1]]])
    
    o1 = n.eval([1, 0])
    o2 = n.eval([0, 1])
    
    print("e:", 2, "a:", o1.index(max(o1)) + 1, "\te:", 4, "a:", o2.index(max(o2)) + 1)
    
    print()
    
    for layers in n.nodes[1:]:
        for node in layers:
            node.print()
            print()
    pass

if __name__ == "__main__":
    main()