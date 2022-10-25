import math
from operator import invert
from turtle import st
import numpy as np
import pandas as pd
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)        
# ======================================================================================= #

def matactivations(act, mat):
    temp = np.zeros(np.shape(mat))
    for row in range(len(mat)):
        for col in range(len(mat[row])):
            # print(row, col, mat[row][col])
            if (mat[row][col] > 10): mat[row][col] = 10 + np.random.uniform(low=-1, high=1)
            elif (mat[row][col] < -10): mat[row][col] = -10 + np.random.uniform(low=-1, high=1)
            temp[row][col] = activations(act, mat[row][col])
    return temp

# ======================================================================================= #

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


def costs(c, act, exp):
    costs = {
        1: diff_squared(act, exp),
        2: None,
        3: None
    }
    return costs[c]


def cost_derivatives(c, act, exp):
    costs = {
        1: diff_squared_derivative(act, exp),
        2: None,
        3: None
    }
    return costs[c]

# ======================================================================================= #

def sigmoid(val):
    return 1 / (1 + math.exp(-val))


def relu(val):
    return max(0, val)


def tangent(val):
    return math.tanh(val)

def softmax(arr):
    r = np.exp(arr) / sum(np.exp(arr))
    # print(r)
    return r

# ======================================================================================= #

def sigmoid_derivative(val):
    return (1 - sigmoid(val)) * sigmoid(val)


def relu_derivative(val):
    out = 0
    if val > 0:
        out = 1
    return out


def tangent_derivative(val):
    return 1 - (math.tanh(val) ** 2)

# ======================================================================================= #

def diff_squared(actual, expected):
    return 0.5 * (actual - expected) ** 2

# ======================================================================================= #

def diff_squared_derivative(actual, expected):
    return (actual - expected)

# ======================================================================================= #

class Network:
    def __init__(self, layers, learning_rate):
        self.error = 100_000_000
        self.l_size = layers[0]
        self.acts = layers[1]
        self.weights = []
        self.biases = []
        self.w_outputs = []
        self.l_outputs = []
        self.l_rate = learning_rate
        self.init()
        pass

    def init(self):
        self.l_outputs.append(np.zeros((self.l_size[0], 1)))
        for layer in range(len(self.l_size) - 1):
            prev = self.l_size[layer]
            curr = self.l_size[layer + 1]
            self.biases.append(np.random.uniform(low=-1.0, high=1.0, size=(curr, 1)))
            self.l_outputs.append(np.zeros((curr, 1)))
            self.w_outputs.append(np.zeros((curr, 1)))
            self.weights.append(np.random.uniform(low=-1.0, high=1.0, size=(curr, prev)))
        pass
    
    def f_prop(self, input_vec):
        in_vec = np.asmatrix(np.array(input_vec)).T
        # self.l_outputs[0] = matactivations(1, in_vec)
        self.l_outputs[0] = in_vec
        for layer in range(1, len(self.l_size)):
            sum = np.matmul(self.weights[layer - 1], np.asmatrix(self.l_outputs[layer - 1]))
            sum = np.add(sum, self.biases[layer - 1])
            self.w_outputs[layer - 1] = sum
            self.l_outputs[layer] = matactivations(self.acts[layer - 1], sum)
        return self.l_outputs[len(self.l_size) - 1]

    def b_prop(self, training_data):
        total_error = np.zeros((len(training_data[1][0]), 1))


        for (input_vec, expected) in zip(training_data[0], training_data[1]):
            actual = self.f_prop(input_vec)
            expected = np.asmatrix(expected).T
            
            output_error = expected - actual # 6.
            tmp = np.array(output_error) * self.l_outputs[len(self.l_outputs) - 1] * (1.0 - self.l_outputs[len(self.l_outputs) - 1]) # 7.
            self.weights[len(self.weights) - 1] = np.add(self.weights[len(self.weights) - 1], self.l_rate * (tmp @ self.l_outputs[len(self.l_outputs) - 2].T)) # 8.
            self.biases[len(self.biases) - 1] = np.add(self.biases[len(self.biases) - 1], self.l_rate * tmp)

            prev_error = output_error
            for layer in range(len(self.weights) - 1, 0, -1):
                if np.array(prev_error).ndim == 1:
                    prev_error = np.reshape(prev_error, (np.size(prev_error), 1))
                curr_error = self.weights[layer].T @ np.array(prev_error) # 9.
                tmp = np.array(curr_error * self.l_outputs[layer] * (1.0 - self.l_outputs[layer])) # 10.
                self.weights[layer - 1] = np.add(self.weights[layer - 1], self.l_rate * (tmp @ self.l_outputs[layer - 1].T)) # 11.
                self.biases[layer - 1] = np.add(self.biases[layer - 1], self.l_rate * tmp)
                prev_error = curr_error.T[0]
                pass
            pass
            total_error += output_error
        # print(total_error)
        total_error = abs(sum(total_error))
        self.error = total_error[0]
        return total_error

    def print(self):
        print(self.weights)
        print(self.biases)
        print(self.w_outputs)
        print(self.l_outputs)

# ======================================================================================= #

def main():
    data = pd.read_csv('./wdbc.data', header=None)
    print('Row: ', len(data), 'Column: ', len(data.columns))
    # print(data)
    input_data = data.iloc[:, 2:]
    output_data = data.iloc[:, :2]
    print(input_data)
    # print(output_data)
    
    for col in range(2, len(input_data.columns)):
        arr = input_data[col].to_numpy().tolist()
        tmp = np.full((len(arr), ), max(arr))
        # arr -= tmp
        x = arr

        x = (x - np.min(x)) / (np.max(x) - np.min(x))

        input_data[col] = x
        
    input_data = input_data.T
    output_data = output_data.T
    
    training_data = [[], []]
    correct, amount = 0, len(data)
    
    for col in range(len(data)):
        # tmp = []
        # arr = softmax((input_data[col].to_numpy()).tolist())
        # for x in range(len(arr)):
        #     # matactivations(1, softmax((input_data[col].to_numpy()).tolist()))
        #     tmp.append(activations(1, arr[x]))
        training_data[0].append(input_data[col].to_numpy().tolist())
        out = [0, 1]
        if output_data[col][1] == 'M':
            out = [1, 0]
        training_data[1].append(out)
    
    for l in training_data[0]:
        a = []
        # print(max(l))
        for n in l:
            b = round(n, 2)
            c = np.format_float_positional(b)
            a.append(float(c))
        # print(max(a), min(a), a)
        
    n = Network([[30, 30, 30, 30, 30, 2], [1, 1, 1, 1, 1]], 0.01)
    tempn = Network([n.l_size, n.acts], n.l_rate)
    tw = n.weights
    tb = n.biases
    # print(tw, tb)
    for (i, o) in zip(training_data[0], training_data[1]):
        out = n.f_prop(i).flatten()
        # print(out, o)
        if (np.argmax(out, axis=0) + 1 == np.argmax(o, axis=0) + 1):
            correct += 1
    perc = correct / amount
    print('Correct: ', str(perc * 100) + '%')
    
    correct = 0
    
    iterations = 50
    epochs = 10
    min_iterations = 200
    min_error = 0.0001
    
    batches = 10
    start = 0
    end = int(int(len(training_data[0]) - 1) / batches)
    gsize = int(int(len(training_data[0]) - 1) / batches)
    for x in range(iterations * epochs * batches * 1 + 1):
        e = n.b_prop([training_data[0][start:end], training_data[1][start:end]])[0]
        start = end % (len(training_data[0]) - 1)
        end = start + gsize
        if (x % 10 == 0):
            # print(start, '\t',  end, '\t', gsize)
            print(x, '\t',  e, '\t', tempn.error)
        if (e >= 0.0000001):
            if e < tempn.error:
                
                tempn.error = n.error
                tempn.weights = n.weights
                tempn.biases = n.biases
            if ((e < min_error) or (tempn.error < min_error)):
                if (x > min_iterations):
                    print("final", e, tempn.error)
                    break
        # elif (e < 0.0000001):
        #     n = Network([n.l_size, n.acts], n.l_rate)
    print(e, '\t', tempn.error)
    
    n.error = tempn.error
    n.weights = tempn.weights
    n.biases = tempn.biases
    # print(np.array_equal(np.array(tw).flatten(), np.array(n.weights).flatten()))
    # print(np.array_equal(np.array(tb).flatten(), np.array(n.biases).flatten()))
        
    for (i, o) in zip(training_data[0], training_data[1]):
        out = n.f_prop(i).flatten()
        # print(out, o)
        if (np.argmax(out, axis=0) + 1 == np.argmax(o, axis=0) + 1):
            correct += 1
        else:
            print(out, o)
    perc = correct / amount
    print('Correct: ', str(perc * 100) + '%')



# ======================================================================================= #

if __name__ == "__main__":
    main()
