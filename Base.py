import math
import numpy as np


def matactivations(act, mat):
    temp = np.zeros(np.shape(mat))
    for row in range(len(mat)):
        for col in range(len(mat[row])):
            temp[row][col] = activations(act, mat[row][col])
    return temp
            


def activations(act, val):
    acts = {
        1: sigmoid(val),
        2: relu(val),
        3: tangent(val)
    }

    return acts[act]


def sigmoid(val):
    return 1 / (1 + math.exp(-val))


def relu(val):
    return max(0, val)


def tangent(val):
    return math.tanh(val)

class Network:
    def __init__(self, layers, learning_rate):
        self.l_size = layers[0]
        self.acts = layers[1]
        self.weights = []
        self.biases = []
        self.w_outputs = []
        self.l_outputs = []
        self.l_rate = learning_rate
        self.init()
        # self.print()
        pass

    def init(self):
        self.l_outputs.append([np.zeros((self.l_size[0], 1))])
        for layer in range(len(self.l_size) - 1):
            prev = self.l_size[layer]
            curr = self.l_size[layer + 1]
            self.biases.append([np.random.uniform(low=-1.0, high=1.0, size=(curr, 1))])
            self.w_outputs.append([np.zeros((curr, 1))])
            self.l_outputs.append([np.zeros((curr, 1))])
            self.weights.append([np.random.uniform(low=-1.0, high=1.0, size=(curr, prev))])
        pass

    def f_prop(self, input_vec):
        in_vec = [np.array(input_vec)]
        self.l_outputs[0] = in_vec
        # self.print()
        for layer in range(1, len(self.l_size)):
            # print(layer - 1, "In: ", self.l_outputs[layer - 1][0])
            # print(layer - 1, "Weights: ", self.weights[layer - 1][0])
            # print(layer - 1, "Biases", self.biases[layer - 1][0])
            # print(layer, "Out: ", self.l_outputs[layer][0])

            # print(self.weights[layer - 1][0], self.l_outputs[layer - 1][0])
            # print(np.shape(self.weights[layer - 1][0]), np.shape(self.l_outputs[layer - 1][0]))

            sum = np.matmul(self.weights[layer - 1][0], self.l_outputs[layer - 1][0])
            sum = np.add(sum, self.biases[layer - 1][0].T)
            self.w_outputs[layer - 1] = sum
            # sum = np.add(self.weights[layer - 1][0] @ self.l_outputs[layer - 1][0], self.biases[layer - 1][0]) # @ means matrix multiplication
            self.l_outputs[layer] = matactivations(self.acts[layer - 1], sum)
            # print()
        return self.l_outputs[len(self.l_size) - 1][0]
        pass

    def b_prop(self, training_data):


        for (input_vec, expected) in zip(training_data[0], training_data[1]):
            actual = self.f_prop(input_vec)

            # output_error = expected - actual # 6.
            # tmp = np.array([output_error * self.l_outputs[2] * (1.0 - self.l_outputs[2])]) # 7.
            # self.weights[1] += self.l_rate * (tmp.T @ self.l_outputs[1]) # 8.

            # hidden_error = self.weights[1][0].T @ np.array([output_error]).T # 9.
            # tmp = np.array([hidden_error * self.l_outputs[1].T * (1.0 - self.l_outputs[1].T)]) # 10.
            # self.weights[0] += self.l_rate * (tmp @ self.l_outputs[0]) # 11.

            error = expected - actual

            for layer in range(len(self.weights) - 1, 0, -1):
                if (layer) % 2 == 0:

                    pass
                else:

                    pass
                pass
            pass
        pass

    def print(self):
        print("Weights:\n", self.weights)
        print("Biases:\n", self.biases)
        print("W_outputs:\n", self.w_outputs)
        print("L_outputs:\n", self.l_outputs)


def main():
    n = Network([[4, 5, 4], [1, 1]], 0.01)
    o1 = n.f_prop([1, 0, 0, 0])
    o2 = n.f_prop([0, 1, 0, 0])
    o3 = n.f_prop([0, 0, 1, 0])
    o4 = n.f_prop([0, 0, 0, 1])
    print(o1, np.argmax(o1, axis=0) + 1, [0, 1, 0, 0])
    print(o2, np.argmax(o2, axis=0) + 1, [0, 0, 1, 0])
    print(o3, np.argmax(o3, axis=0) + 1, [0, 0, 0, 1])
    print(o4, np.argmax(o4, axis=0) + 1, [1, 0, 0, 0])
    print()
    # n.print()
    val = 100000
    for x in range(val):
        n.b_prop([[[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]], [[0,1,0,0], [0,0,1,0], [0,0,0,1], [1,0,0,0]]])
    o1 = n.f_prop([1, 0, 0, 0])
    o2 = n.f_prop([0, 1, 0, 0])
    o3 = n.f_prop([0, 0, 1, 0])
    o4 = n.f_prop([0, 0, 0, 1])
    print(o1, np.argmax(o1, axis=0) + 1, [0, 1, 0, 0])
    print(o2, np.argmax(o2, axis=0) + 1, [0, 0, 1, 0])
    print(o3, np.argmax(o3, axis=0) + 1, [0, 0, 0, 1])
    print(o4, np.argmax(o4, axis=0) + 1, [1, 0, 0, 0])



if __name__ == "__main__":
    main()
