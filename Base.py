import math
import numpy as np
import pandas as pd

# =========================================== Activation on matrix =========================================== #

# Function to apply an activation
def matactivations(act, mat):
    temp = np.zeros(np.shape(mat))
    for row in range(len(mat)):
        for col in range(len(mat[row])):
            if (mat[row][col] > 10): mat[row][col] = 10 + np.random.uniform(low=-1, high=1)
            elif (mat[row][col] < -10): mat[row][col] = -10 + np.random.uniform(low=-1, high=1)
            temp[row][col] = activations(act, mat[row][col])
    return temp

def matcosts_derivatives(act, a, e):
    temp = np.zeros(np.shape(a))
    for row in range(len(a)):
        for col in range(len(a[row])):
            temp[row][col] = cost_derivatives(act, a[row][col], e[row][col])
    return temp

# =========================================== Activation on value =========================================== #

# Function for each activation to be called from
def activations(act, val):
    acts = {
        1: sigmoid(val),
        2: relu(val),
        3: tangent(val)
    }

    return acts[act]

# Function of activation derivatives
def activations_derivatives(act, val):
    acts = {
        1: sigmoid_derivative(val),
        2: relu_derivative(val),
        3: tangent_derivative(val)
    }
    return acts[act]

# Function for cost functions
def costs(c, act, exp):
    costs = {
        1: diff_squared(act, exp),
        2: None,
        3: None
    }
    return costs[c]

# Function of cost derivatives
def cost_derivatives(c, act, exp):
    costs = {
        1: diff_squared_derivative(act, exp),
        2: None,
        3: None
    }
    return costs[c]

# =========================================== Activation Functions =========================================== #

def sigmoid(val):
    return 1 / (1 + math.exp(-val))


def relu(val):
    return max(0, val)


def tangent(val):
    return math.tanh(val)

def softmax(arr):
    r = np.exp(arr) / sum(np.exp(arr))
    return r

# =========================================== Activation Derivatives =========================================== #

def sigmoid_derivative(val):
    return (1 - sigmoid(val)) * sigmoid(val)


def relu_derivative(val):
    out = 0
    if val > 0:
        out = 1
    return out


def tangent_derivative(val):
    return 1 - (math.tanh(val) ** 2)

# =========================================== Cost Functions =========================================== #

def diff_squared(actual, expected):
    return 0.5 * (actual - expected) ** 2

# =========================================== Cost Derivatives =========================================== #

def diff_squared_derivative(actual, expected):
    return (actual - expected)

# =========================================== Class =========================================== #

class Network:
    def __init__(self, layers, learning_rate):
        # initialises all class variables
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
        # create matrices
        self.l_outputs.append(np.zeros((self.l_size[0], 1))) # add 1D input layer matrix
        for layer in range(len(self.l_size) - 1):
            prev = self.l_size[layer]
            curr = self.l_size[layer + 1]
            self.biases.append(np.random.uniform(low=-1.0, high=1.0, size=(curr, 1))) # add 1D bias layer matrix
            self.l_outputs.append(np.zeros((curr, 1))) # add 1D output matrix
            self.w_outputs.append(np.zeros((curr, 1))) # add 1D weighted output matrix
            self.weights.append(np.random.uniform(low=-1.0, high=1.0, size=(curr, prev))) # add 2D weights matrix
        pass
    
    def f_prop(self, input_vec):
        in_vec = np.asmatrix(np.array(input_vec)).T # convert input vector to matrix
        self.l_outputs[0] = in_vec
        for layer in range(1, len(self.l_size)):
            sum = np.matmul(self.weights[layer - 1], np.asmatrix(self.l_outputs[layer - 1])) # multiply weights by the prev layer outputs
            sum = np.add(sum, self.biases[layer - 1]) # add bias for current layer to sum
            self.w_outputs[layer - 1] = sum # set weighted output for layer
            self.l_outputs[layer] = matactivations(self.acts[layer - 1], sum) # apply activation function to weighted sum
        return self.l_outputs[len(self.l_size) - 1] # return output layer matrix

    def b_prop(self, training_data):
        total_error = np.zeros((len(training_data[1][0]), 1)) # create matrix of zeros

        for (input_vec, expected) in zip(training_data[0], training_data[1]):
            actual = self.f_prop(input_vec) # calculate output for input vector
            expected = np.asmatrix(expected).T
            
            output_error = expected - actual # calculate error of each output in final layer
            tmp = np.array(output_error) * self.l_outputs[len(self.l_outputs) - 1] * (1.0 - self.l_outputs[len(self.l_outputs) - 1]) # scale error by previous layer outputs
            self.weights[len(self.weights) - 1] = np.add(self.weights[len(self.weights) - 1], self.l_rate * (tmp @ self.l_outputs[len(self.l_outputs) - 2].T)) # change weights with respect to learning rate and error
            self.biases[len(self.biases) - 1] = np.add(self.biases[len(self.biases) - 1], self.l_rate * tmp) # change biases with respect to learning rate and error

            prev_error = output_error
            for layer in range(len(self.weights) - 1, 0, -1):
                if np.array(prev_error).ndim == 1:
                    prev_error = np.reshape(prev_error, (np.size(prev_error), 1)) # make sure error is a single column matrix
                curr_error = self.weights[layer].T @ np.array(prev_error) # propogate error back with respect to previous layer weights
                tmp = np.array(curr_error * self.l_outputs[layer] * (1.0 - self.l_outputs[layer])) # scale error by previous layer outputs
                self.weights[layer - 1] = np.add(self.weights[layer - 1], self.l_rate * (tmp @ self.l_outputs[layer - 1].T)) # change weights with respect to learning rate and error
                self.biases[layer - 1] = np.add(self.biases[layer - 1], self.l_rate * tmp) # change biases with respect to learning rate and error
                prev_error = curr_error.T[0] # set previous error to current error
                pass
            pass
            total_error += output_error # add output error to total error
        total_error = abs(sum(total_error)) # sum total error and make positive
        self.error = total_error[0] # set networks error
        return total_error # return total network error

    def print(self):
        print(self.weights)
        print(self.biases)
        print(self.w_outputs)
        print(self.l_outputs)

# =========================================== Main =========================================== #

def main():
    # import data using pandas
    data = pd.read_csv('./wdbc.data', header=None)
    # split data into input and output
    input_data = data.iloc[:, 2:]
    output_data = data.iloc[:, :2]
    
    # normalize each column to between 0 and 1
    # formula for normalization taken from
    # https://www.geeksforgeeks.org/how-to-normalize-an-numpy-array-so-the-values-range-exactly-between-0-and-1/
    for col in range(2, len(input_data.columns)):
        x = input_data[col].to_numpy().tolist() # convert pandas column to list
        x = (x - np.min(x)) / (np.max(x) - np.min(x)) # normalize column
        input_data[col] = x # return normalized list to pandas data frame

    # Transpose input and output data to access columns
    input_data = input_data.T
    output_data = output_data.T
    
    # initialise training data array and calculation variables
    training_data = [[], []]
    correct, amount = 0, len(data)
    
    # add input data and output data to training data
    for col in range(len(data)):
        training_data[0].append(input_data[col].to_numpy().tolist())
        out = [0, 1]
        # check if ouput should be M or B
        if output_data[col][1] == 'M':
            # change output for training data if output is M
            out = [1, 0]
        # add to training data to match with input data
        training_data[1].append(out)
    
    input_layers = []
    activation_values = [1, 2, 3]
    activations = []
    
    # define network parameters
    layers, learning_rate, iterations, epochs, min_iterations, min_error, batches, val = 0, 0, 0, 0, 0, 0, 0, 0
    print("Parameters used during testing:")
    print("\n", "Layers: 3\n", "Learning Rate: 0.01\n", "Nodes: 30, 20, 2\n", "Activations: 1, 1, 1\n", "Iterations: 100\n", "Epochs: 10\n", "Minimum Iterations: 200\n", "Minimum Error: 0.0001\n", "Batches: 10\n")
    print("")
    
    print("If using a .txt file please layout as above ^^ description without headers, plese have file in current directory, named: NeuralNetwork_Inputs.txt. Example below")
    print("\n", "3\n", "0.01\n", "30, 20, 2\n", "1, 1, 1\n", "100\n", "10\n", "200\n", "0.0001\n", "10\n")

    if (int(input("Would you like to input from a text file? (1 if No, 0 if Yes):\n>> "))):
        while (layers < 2): layers = int(input("How many layers will this network have? (default: 2)\n>> ") or 2)
        while (learning_rate <= 0): learning_rate = float(input("What is the learning rate of this network? (default: 0.01)\n>> ") or  0.01)
        print("1: Sigmoid\n2: Relu\n3: Tangent")
        for l in range(layers):
            while (val < 1):
                val = int(input("How many nodes in layer {0}? (default: 1)\n>> ".format(l + 1)) or 1)
                input_layers.append(val)
            val = 0
            while (val not in activation_values):
                val = int(input("Which activation would you like for layer {0}? (default: 1)\n>> ".format(l + 1)) or 1)
                activations.append(val)
                print(val in activation_values)
            val = 0
            
        # define training parameters
        while (iterations <= 0): iterations = int(input("How many Iterations of training would you like? (default: 50)\n>> ") or 50)
        while (epochs <= 0): epochs = int(input("How many Epochs would you like? (default: 10)\n>> ") or 10)
        while (min_iterations <= 0): min_iterations = int(input("Minimum iterations of learning? (default: 200)\n>> ") or 200)
        while (min_error <= 0): min_error = float(input("Minimum error of network? (default: 0.0005)\n>> ") or 0.0005)
        
        while (batches < 1): batches = int(input("How many Batches of data? (default: 1)\n>> ") or 1)
    else:
        f = open("NeuralNetwork_Inputs.txt", "r")
        ins = f.readlines()
        temp = []
        for (s, i) in zip(ins, range(len(ins))):
            s = s.replace("\n", "")
            s = s.replace(",", "")
            if i == 2 or i == 7:
                temp.append([float(i) for i in s.split()])
            else:
                temp.append([int(i) for i in s.split()])

            if len(temp[i]) == 1:
                temp[i] = temp[i][0]
        input_layers = temp[1]
        activations = temp[3]
        learning_rate = temp[2]
        iterations = temp[4]
        epochs = temp[5]
        min_iterations = temp[6]
        min_error = temp[7]
        batches = temp[8]
       
    # create two networks
    n = Network([input_layers, activations], learning_rate)
    tempn = Network([n.l_size, n.acts], n.l_rate)

    # calculate how many outputs the network has gotten correct with no training
    for (i, o) in zip(training_data[0], training_data[1]):
        out = n.f_prop(i).flatten()
        if (np.argmax(out, axis=0) + 1 == np.argmax(o, axis=0) + 1):
            correct += 1
    perc = correct / amount
    print('Correct: ', str(perc * 100) + '%')
    
    correct = 0 # reset correct variable
    
    start = 0
    end = int(int(len(training_data[0]) - 1) / batches)
    bsize = int(int(len(training_data[0]) - 1) / batches)
    
    finished = False
    # start training loop
    for epoch in range(epochs):
        if finished: break
        for x in range(iterations):
            if finished: break
            for batch in range(batches):
                if finished: break
                dat = [training_data[0][start:end], training_data[1][start:end]] # calculate training data based on batch data
                e = n.b_prop(dat)[0] # back prop dat
                
                # update start and end of batch training array
                start = end % (len(training_data[0]) - 1)
                end = start + bsize
                
                it = epoch * epochs * iterations + x * batches + batch
                if (it % 10 == 0):
                    # print(start, '\t',  end, '\t', gsize)
                    print(it, '\t',  e, '\t', tempn.error)
                # if error is greater than 1*10^-7
                if (e >= 0.0000001):
                    if e < tempn.error: # if error is smaller than temp network error
                        # copy current network parameters to temp network
                        tempn.error = n.error
                        tempn.weights = n.weights
                        tempn.biases = n.biases
                    # if error is below minimum error value and the network has been through the minimum number of iterations
                    if ((tempn.error < min_error) and (it > min_iterations)):
                        # break out of training loop
                        print("final", e, tempn.error)
                        finished = True
                elif (e < 0.0000001):
                    # create new network if error too low
                    n = Network([n.l_size, n.acts], n.l_rate)
    print(e, '\t', tempn.error)
    
    n.error = tempn.error
    n.weights = tempn.weights
    n.biases = tempn.biases
    
    # calculate how many outputs the network has gotten correct after training
    for (i, o) in zip(training_data[0], training_data[1]):
        out = n.f_prop(i).flatten()
        if (np.argmax(out, axis=0) + 1 == np.argmax(o, axis=0) + 1):
            correct += 1
        else:
            # print(out, o)
            pass
    perc = correct / amount
    print('Correct: ', str(perc * 100) + '%')



# ===========================================  =========================================== #

if __name__ == "__main__":
    main()
