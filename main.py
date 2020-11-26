import numpy as np


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
        # self.bias = np.random.normal()

    def get_input(self, inputs):
        return np.dot(self.weights, inputs)

    def feedforward(self, inputs):
        # np.dot - вычисляем скалярное произведение массивов
        return sigmoid(self.get_input(inputs))

    def train(self, input_neuron_outputs, self_d, learn_rate):

        GRADs = np.array([input_neuron_output * self_d for input_neuron_output in input_neuron_outputs])
        self.weights += GRADs * learn_rate
        input_neuron_ds = np.array([weight * self_d * deriv_sigmoid(input_neuron_output)
                                    for weight, input_neuron_output
                                    in zip(self.weights, input_neuron_outputs)])

        return np.array(input_neuron_ds)


class Layer:
    def __init__(self, neuron_number, weights_amount):
        self.neurons = [Neuron(weights_amount) for i in range(neuron_number)]

    def size(self):
        return len(self.neurons)

    def feedforward(self, inputs):
        return np.array([neuron.feedforward(inputs) for neuron in self.neurons])

    def train(self, input_neuron_outputs, neuron_ds, learn_rate):
        res = np.array([0.0 for i in range(len(self.neurons))])
        for neuron, neuron_d in zip(self.neurons, neuron_ds):
            res += neuron.train(input_neuron_outputs=input_neuron_outputs,
                                self_d=neuron_d,
                                learn_rate=learn_rate)
        return res


class NeuralNetwork:
    def __init__(self):
        # weights = np.array([0, 1])
        # bias = 0

        self.h = Layer(2, 2)

        self.g = Layer(2, 2)

        self.o = Neuron(2)

    def feedforward(self, x):
        out_h = self.h.feedforward(x)
        out_g = self.g.feedforward(out_h)
        out_o = self.o.feedforward(out_g)

        return out_o

    def train(self, data, y_trues, epochs=1000, learn_rate=0.01):
        for epoch in range(epochs):
            for inputs, y_true in zip(data, y_trues):

                h_output = self.h.feedforward(inputs)
                g_output = self.g.feedforward(h_output)
                o_output = self.o.feedforward(g_output)

                d_o = (y_true - o_output) * deriv_sigmoid(o_output)

                d_gs = self.o.train(input_neuron_outputs=g_output,
                                    self_d=d_o,
                                    learn_rate=learn_rate)

                d_hs = self.g.train(input_neuron_outputs=h_output,
                                    neuron_ds=d_gs,
                                    learn_rate=learn_rate)

                # print(d_o, d_gs, d_hs, sep='\n')

                self.h.train(input_neuron_outputs=inputs,
                             neuron_ds=d_hs,
                             learn_rate=learn_rate)

            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))


# Определение набора данных
data = np.array([
    [-2, -1],  # Alice
    [25, 6],  # Bob
    [17, 4],  # Charlie
    [-15, -6],  # Diana
])

# data = np.array([[133, 65], [160, 72], [152, 70], [120, 60]])

all_y_trues = np.array([
    1,  # Alice
    0,  # Bob
    0,  # Charlie
    1,  # Diana
])

# Тренируем нашу нейронную сеть!
network = NeuralNetwork()
network.train(data, all_y_trues, 5000, 0.1)

# Make some predictions
emily = np.array([-7, -3])# 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches

# emily = np.array([128, 63]) # 128 pounds, 63 inches
# frank = np.array([155, 68])  # 155 pounds, 68 inches

emily_res = network.feedforward(emily)
frank_res = network.feedforward(frank)


def number_to_sex(number):
    if number >= 0.5:
        return 'F'
    return 'M'


print("Emily: %.3f" % emily_res, number_to_sex(emily_res))
print("Frank: %.3f" % frank_res, number_to_sex(frank_res))