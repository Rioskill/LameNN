import numpy as np
import sys
import json

from paint import Paint

alpha = 0.1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))   # np.exp - считаем экспоненту


def deriv_sigmoid(x):
    # Производная от sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    # y_true и y_pred являются массивами numpy с одинаковой длиной
    return ((y_true - y_pred) ** 2).mean()


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

        # self.last_deltas = np.array([0 for i in range(weights_amount)])

    def get_input(self, inputs):
        return np.dot(self.weights, inputs) # + self.bias

    def feedforward(self, inputs):
        # np.dot - вычисляем скалярное произведение массивов
        return sigmoid(self.get_input(inputs))

    def train(self, input_neuron_outputs, self_d, learn_rate):
        input_sum = deriv_sigmoid(self.get_input(input_neuron_outputs))
        self.weights += learn_rate * input_neuron_outputs * self_d * input_sum
        input_neuron_ds = np.array(self.weights) * self_d * input_sum
        return np.array(input_neuron_ds)


def create_neuron(weights_amount):
    weights = np.array([np.random.normal() for i in range(weights_amount)])
    bias = np.random.normal()
    return Neuron(weights, bias)


class Layer:
    def __init__(self, neurons):
        self.neurons = neurons

    def size(self):
        return len(self.neurons)

    def feedforward(self, inputs):
        return np.array([neuron.feedforward(inputs) for neuron in self.neurons])

    def train(self, input_neuron_outputs, neuron_ds, learn_rate):
        d_previous_layer = np.array([0.0 for i in range(len(input_neuron_outputs))])
        for neuron, neuron_d in zip(self.neurons, neuron_ds):
            d_previous_layer += neuron.train(input_neuron_outputs=input_neuron_outputs,
                                             self_d=neuron_d,
                                             learn_rate=learn_rate)
        return d_previous_layer

    def get_data(self):
        return [(list(neuron.weights), neuron.bias) for neuron in self.neurons]


def create_layer(weights_amount, neuron_number):
    neurons = [create_neuron(weights_amount) for i in range(neuron_number)]
    return Layer(neurons)


class NeuralNetwork:
    def __init__(self, input_layer_size, layers, output_layer_size):
        self.input_layer_size = input_layer_size
        self.hidden_layers = layers

        self.output_layer = create_layer(self.hidden_layers[-1].size(), output_layer_size)

    def save(self, path):
        hidden_layers = [layer.get_data() for layer in self.hidden_layers]
        output = {'input_layer_size': self.input_layer_size,
                  'hidden_layers': hidden_layers,
                  'output_layer': self.output_layer.get_data()}
        output = json.dumps(output)

        with open(path, 'w') as file:
            file.write(output)

    def feedforward(self, x):
        out = np.array(x)
        for layer in self.hidden_layers:
            out = layer.feedforward(out)
        out_o = self.output_layer.feedforward(out)

        return out_o

    def train(self, data, y_trues, epochs=1000, learn_rate=0.01):
        for epoch in range(epochs):

            i = 1
            data_length = len(data)
            S_length = 0
            for inputs, y_true in zip(data, y_trues):
                S = f"{epoch + 1} epoch; {i}/{data_length} image"
                sys.stdout.write('\b' * S_length + S)
                S_length = len(S)
                i += 1

                outputs = [inputs]
                for layer in self.hidden_layers:
                    outputs.append(layer.feedforward(outputs[-1]))

                o_output = self.output_layer.feedforward(outputs[-1])

                d_o = 2 * (y_true - o_output) * deriv_sigmoid(o_output)

                d_layer = (self.output_layer.train(input_neuron_outputs=outputs[-1],
                                                   neuron_ds=d_o,
                                                   learn_rate=learn_rate))

                for layer, input_neuron_output in zip(self.hidden_layers[::-1], outputs[-2::-1]):
                    d_layer = layer.train(input_neuron_outputs=input_neuron_output,
                                          neuron_ds=d_layer,
                                          learn_rate=learn_rate)

            sys.stdout.write('\b' * S_length)
            y_preds = np.apply_along_axis(self.feedforward, 1, data)
            loss = mse_loss(y_trues, y_preds)
            sys.stdout.write("Epoch %d loss: %.3f" % (epoch + 1, loss) + '\n')


def make_network(input_size, layers_sizes, output_size):
    neuron_layers = list()
    for size in layers_sizes:
        neuron_layers.append(create_layer(input_size, size))
        input_size = size
    return NeuralNetwork(input_size, neuron_layers, output_size)


def read_network(path):
    with open(path, 'r') as file:
        input = json.loads(file.read())
        input_layer_size = input['input_layer_size']
        hidden_layers_data = input['hidden_layers']

        hidden_layers = [Layer([Neuron(neuron_data[0], neuron_data[1]) for neuron_data in layer]) for layer in hidden_layers_data]

        output_layer_data = input['output_layer']
        output_layer = Layer([Neuron(neuron_data[0], neuron_data[1]) for neuron_data in output_layer_data])

        network = NeuralNetwork(input_layer_size, hidden_layers, output_layer.size())
        network.output_layer = output_layer
        return network


data_file = open('./mnist_test/mnist_train.csv','r')
training_list = data_file.readlines()[:1000]
data_file.close()


dataset = list()
trues = list()
print("making dataset")

for record in training_list:
    all_values = record.split(',')
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = np.zeros(10) + 0.01
    targets[int(all_values[0])] = 0.99

    dataset.append(inputs)
    trues.append(targets)

dataset = np.array(dataset)
trues = np.array(trues)

network = make_network(784, [40, 40], 10)

print("training network")

network.train(dataset, trues, 5, 0.3)

print("training done")

network.save('saves/second')

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

second_network = read_network('saves/second')

for my_network in (network, second_network):
    right_guesses = 0
    all_guesses = len(test_lines)

    for line in test_lines:
        arr = line.split(',')
        target = int(arr[0])
        inputs = np.asfarray(arr[1:])
        guess = my_network.feedforward(inputs)
        guessed_number = get_number(guess)

        if guessed_number == target:
            right_guesses += 1


    print(str(right_guesses) + '/' + str(all_guesses), str(right_guesses / all_guesses * 100) + '%')
