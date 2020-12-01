import numpy as np
import sys
from itertools import repeat
from numba import njit, float64
from numba import cuda
import math

from paint import Paint


alpha = 0.1


@njit
def sigmoid(x):
    return 1 / (1 + np.exp(-x))   # np.exp - считаем экспоненту


@njit
def deriv_sigmoid(x):
    # Производная от sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


@cuda.jit(device=True)
def sigmoid_device(x):
    return 1 / (1 + math.e ** (-x))


@cuda.jit(device=True)
def deriv_sigmoid_device(x):
    fx = sigmoid_device(x)
    return fx * (1 - fx)


@njit
def mse_loss(y_true, y_pred):
    # y_true и y_pred являются массивами numpy с одинаковой длиной
    return ((y_true - y_pred) ** 2).mean()


class Neuron:
    def __init__(self, weights_amount):
        self.weights = np.array([np.random.normal() for i in range(weights_amount)])
        # self.bias = np.random.normal()

        self.last_deltas = np.array([0.0 for i in range(weights_amount)])

    def get_input(self, inputs):
        return np.dot(self.weights, inputs) # + self.bias

    def feedforward(self, inputs):
        return sigmoid(self.get_input(inputs))

    def train(self, input_neuron_outputs, self_d, learn_rate):

        GRADs = np.array([input_neuron_output * self_d for input_neuron_output in input_neuron_outputs])

        self.last_deltas = GRADs * learn_rate + alpha * self.last_deltas
        self.weights += self.last_deltas

        # biasGRAD = self.bias * self_d
        # self.bias += biasGRAD * learn_rate

        input_neuron_ds = np.array([weight * self_d * deriv_sigmoid(input_neuron_output)
                                    for weight, input_neuron_output
                                    in zip(self.weights, input_neuron_outputs)])

        return input_neuron_ds


@cuda.jit(debug=True)
def train_layer(weights, input_neuron_ds, input_neuron_outputs, neuron_ds, last_deltas, learn_rate):
    i, j = cuda.grid(2)
    # for i in range(len(weights)):
    if i < weights.shape[0]:
        # for j in range(len(input_neuron_outputs)):
        if j < weights.shape[1]:
            GRAD = input_neuron_outputs[j] * neuron_ds[i]
            last_deltas[i][j] = GRAD * learn_rate + alpha * last_deltas[i][j]
            weights[i][j] += last_deltas[i][j]

            input_neuron_ds[j] = weights[i][j] * neuron_ds[i] * deriv_sigmoid_device(input_neuron_outputs[j])


class Layer:
    def __init__(self, weights_amount, neuron_number):
        self.neurons = [Neuron(weights_amount) for i in range(neuron_number)]

        self.device_weights = cuda.to_device(np.array([neuron.weights for neuron in self.neurons]))
        self.device_last_deltas = cuda.to_device(np.array([neuron.last_deltas for neuron in self.neurons]))

    def size(self):
        return len(self.neurons)

    def feedforward(self, inputs):
        return np.array([neuron.feedforward(inputs) for neuron in self.neurons])

    def train(self, input_neuron_outputs, neuron_ds, learn_rate):
        # d_previous_layer = np.array([0.0 for i in range(len(input_neuron_outputs))])
        # for neuron, neuron_d in zip(self.neurons, neuron_ds):
        #     d_previous_layer += neuron.train(input_neuron_outputs,
        #                                      neuron_d,
        #                                      learn_rate)
        #
        # return d_previous_layer

        d_previous_layer_device = cuda.to_device(np.zeros(shape=(len(input_neuron_outputs))))


        # d_previous_layer = np.array([0.0 for i in range(len(input_neuron_outputs))])

        threads_per_block = (4, 4)

        blocks_per_grid_x = math.ceil(len(self.neurons) / threads_per_block[0])
        blocks_per_grid_y = math.ceil(len(input_neuron_outputs) / threads_per_block[1])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        # print(self.device_weights.shape, (blocks_per_grid[0] * threads_per_block[0], blocks_per_grid[1] * threads_per_block[1]))

        # weights = np.array([np.array(neuron.weights) for neuron in self.neurons])
        # last_deltas = np.array([np.array(neuron.last_deltas) for neuron in self.neurons])

        train_layer[blocks_per_grid, threads_per_block](
            self.device_weights,
            d_previous_layer_device,
            input_neuron_outputs,
            neuron_ds,
            self.device_last_deltas,
            learn_rate)

        cuda.synchronize()


        last_deltas = self.device_last_deltas.copy_to_host()
        weights = self.device_weights.copy_to_host()

        # print(len(neuron_ds),  weights.shape)

        for i in range(len(self.neurons)):
            self.neurons[i].last_deltas = last_deltas[i]
            self.neurons[i].weights = weights[i]

        self.device_weights = cuda.to_device(np.array([neuron.weights for neuron in self.neurons]))
        self.device_last_deltas = cuda.to_device(np.array([neuron.last_deltas for neuron in self.neurons]))

        d_previous_layer = d_previous_layer_device.copy_to_host()

        return d_previous_layer


class NeuralNetwork:
    def __init__(self, input_layer_size, layers, output_layer_size):
        self.input_layer_size = input_layer_size
        self.hidden_layers = layers

        # self.progress_bar = ProgressBar(20, 100)

        # self.o = Neuron(self.hidden_layers[-1].size())

        self.o = Layer(self.hidden_layers[-1].size(), output_layer_size)

    def feedforward(self, x):
        out = np.array(x)
        for layer in self.hidden_layers:
            out = layer.feedforward(out)
        out_o = self.o.feedforward(out)

        return out_o

    def train(self, data, y_trues, epochs=1000, learn_rate=0.01):
        # self.progress_bar.begin()
        for epoch in range(epochs):

            for inputs, y_true in zip(data, y_trues):

                outputs = [inputs]
                for layer in self.hidden_layers:
                    outputs.append(layer.feedforward(outputs[-1]))

                o_output = self.o.feedforward(outputs[-1])

                d_o = list()
                for output, true in zip(o_output, y_true):
                    d_o.append((true - output) * deriv_sigmoid(output))

                d_o = np.array(d_o)

                d_layer = self.o.train(input_neuron_outputs=outputs[-1],
                                       neuron_ds=d_o,
                                       learn_rate=learn_rate)

                for layer, input_neuron_output in zip(self.hidden_layers[::-1], outputs[-2::-1]):
                    d_layer = layer.train(input_neuron_outputs=input_neuron_output,
                                          neuron_ds=d_layer,
                                          learn_rate=learn_rate)

            # self.progress_bar.update()

            # if epoch % 100 == 0:
            if True:
                # self.progress_bar.close()
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))

                # self.progress_bar.begin()
        # self.progress_bar.close()


def make_network(input_size, layers_sizes, output_size):
    neuron_layers = list()
    for size in layers_sizes:
        neuron_layers.append(Layer(input_size, size))
        input_size = size
    return NeuralNetwork(input_size, neuron_layers, output_size)


# data = np.array([[133, 65], [160, 72], [152, 70], [120, 60]])
#
# all_y_trues = np.array([
#     [1],  # Alice
#     [0],  # Bob
#     [0],  # Charlie
#     [1],  # Diana
# ])
#
# # Тренируем нашу нейронную сеть!
# network = make_network(2, [20, 20], 1)
# network.train(data, all_y_trues, 2000, 0.01)
# quit()


data_file = open('./mnist_test/mnist_train.csv','r')
training_list = data_file.readlines()
data_file.close()


dataset = list()
trues = list()
print("making dataset")

for record in training_list[:1000]:
    all_values = record.split(',')
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = np.zeros(10) + 0.01
    targets[int(all_values[0])] = 0.99

    dataset.append(inputs)
    trues.append(targets)

dataset = np.array(dataset)
trues = np.array(trues)

network = make_network(784, [20, 20], 10)

print("training network")

network.train(dataset, trues, 50, 0.1)

print("training done")


def get_number(network_output):
    max_num = 0
    max_pos = -1
    for i, number in enumerate(network_output):
        if number > max_num:
            max_num = number
            max_pos = i
    return max_pos


test_file = open('./mnist_test/mnist_test.csv', 'r')
test_lines = test_file.readlines()
test_file.close()

right_guesses = 0
all_guesses = len(test_lines)

for line in test_lines:
    arr = line.split(',')
    target = int(arr[0])
    inputs = np.asfarray(arr[1:])
    guess = network.feedforward(inputs)
    guessed_number = get_number(guess)

    # print(guessed_number, target, guessed_number == target)
    if guessed_number == target:
        right_guesses += 1

    # print("res: " + str(get_number(res)), res, "answer: " + str(target), sep="\n")

print(str(right_guesses) + '/' + str(all_guesses), str(right_guesses / all_guesses * 100) + '%')


def evaluate_drawing(data):
    print(get_number(network.feedforward(data)))


paint = Paint(evaluate_drawing)
paint.window.mainloop()
