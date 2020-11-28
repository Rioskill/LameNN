import numpy as np
import sys


class ProgressBar:
    def __init__(self, real_width, data_width):
        self.width = real_width
        self.data_batch_width = data_width // real_width
        self.it = 0

    def begin(self):
        self.it = 0
        sys.stdout.write("[%s]" % (" " * self.width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.width + 1))

    def update(self):
        if self.it % self.data_batch_width == 0:
            sys.stdout.write("-")

            number_of_lines = int(self.it // self.data_batch_width)
            number_of_spaces = self.width - number_of_lines - 1

            counter = '[' + str(self.it + self.data_batch_width) + '/' + str(self.width * self.data_batch_width) + ']'

            sys.stdout.write(" " * number_of_spaces + ']')
            sys.stdout.write("\b" * (self.width - number_of_lines + len(counter)))
            sys.stdout.flush()
        self.it += 1

    def close(self):
        sys.stdout.write("]\n")  # this ends the progress bar


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
    def __init__(self, weights_amount):
        self.weights = np.array([np.random.normal() for i in range(weights_amount)])
        self.bias = np.random.normal()

        self.last_deltas = np.array([0 for i in range(weights_amount)])

    def get_input(self, inputs):
        return np.dot(self.weights, inputs) # + self.bias

    def feedforward(self, inputs):
        # np.dot - вычисляем скалярное произведение массивов
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

        return np.array(input_neuron_ds)


def train_neuron(data):
    neuron = data[0]
    input_neuron_outputs = data[1]
    neuron_d = data[2]
    learn_rate = data[3]

    return neuron.train(input_neuron_outputs=input_neuron_outputs,
                        self_d=neuron_d,
                        learn_rate=learn_rate)


class Layer:
    def __init__(self, weights_amount, neuron_number):
        self.neurons = [Neuron(weights_amount) for i in range(neuron_number)]

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



        # d_previous_layer = sum(pool.map(train_neuron, zip(self.neurons, [input_neuron_outputs] * len(self.neurons), neuron_ds, [learn_rate] * len(self.neurons))))
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

                d_layer = (self.o.train(input_neuron_outputs=outputs[-1],
                                        neuron_ds=d_o,
                                        learn_rate=learn_rate))

                for layer, input_neuron_output in zip(self.hidden_layers[::-1], outputs[-2::-1]):
                    d_layer = layer.train(input_neuron_outputs=input_neuron_output,
                                          neuron_ds=d_layer,
                                          learn_rate=learn_rate)

            # self.progress_bar.update()

            #if epoch % 1 == 0:
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
# # F M
# all_y_trues = np.array([
#     np.array([1, 0]),  # Alice
#     np.array([0, 1]),  # Bob
#     np.array([0, 1]),  # Charlie
#     np.array([1, 0]),  # Diana
# ])
#
# # Тренируем нашу нейронную сеть!
# network = make_network(2, [20, 20], 2)
# network.train(data, all_y_trues, 5000, 0.005)
#
# emily = np.array([128, 63])  # 128 pounds, 63 inches
# frank = np.array([155, 68])  # 155 pounds, 68 inches
#
# emily_res = network.feedforward(emily)
# frank_res = network.feedforward(frank)
#
# print(emily_res)
# print(frank_res)


data_file = open('./mnist_test/mnist_train.csv','r')
training_list = data_file.readlines()
data_file.close()

dataset = list()
trues = list()
print("making dataset")

for record in training_list[:100]:
    all_values = record.split(',')
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = np.zeros(10) + 0.01
    targets[int(all_values[0])] = 0.99

    dataset.append(inputs)
    trues.append(targets)

dataset = np.array(dataset)
trues = np.array(trues)

network = make_network(784, [10], 10)

print("training network")
network.train(dataset, trues, 20, 0.1)

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
test_lines = test_file.readlines()[:50]
test_file.close()

for line in test_lines:
    arr = line.split(',')
    target = arr[0]
    inputs = np.asfarray(arr[1:])
    res = network.feedforward(inputs)
    print("res: " + str(get_number(res)), res, "answer: " + str(target), sep="\n")
