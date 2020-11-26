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
    def __init__(self, weights=None, bias=None):
        self.weights = weights   # веса
        self.bias = bias         # смещение

        if self.weights is None:
            self.weights = np.array([np.random.normal(), np.random.normal()])
        if self.bias is None:
            self.bias = np.random.normal()

    def get_input(self, inputs):
        return np.dot(self.weights, inputs) + self.bias

    def feedforward(self, inputs):
        # np.dot - вычисляем скалярное произведение массивов
        return sigmoid(self.get_input(inputs))


class NeuralNetwork:

    def __init__(self):
        # weights = np.array([0, 1])
        # bias = 0

        self.h1 = Neuron()
        self.h2 = Neuron()

        self.o = Neuron()

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        out_o = self.o.feedforward(np.array([out_h1, out_h2]))

        return out_o

    def train(self, data, y_trues, epochs=1000):
        learn_rate = 0.1

        for epoch in range(epochs):
            for inputs, y_true in zip(data, y_trues):

                h1_input = self.h1.get_input(inputs)
                h1_output = sigmoid(h1_input)

                h2_input = self.h2.get_input(inputs)
                h2_output = sigmoid(h2_input)

                o_inputs = np.array([h1_output, h2_output])
                o_input = self.o.get_input(o_inputs)
                o_output = sigmoid(o_input)

                error = ((y_true - o_output) ** 2)

                d_o = (y_true - o_output) * deriv_sigmoid(o_output)

                w5, w6 = self.o.weights

                d_h1 = w5 * d_o * deriv_sigmoid(h1_output)

                d_h2 = w6 * d_o * deriv_sigmoid(h2_output)

                GRAD_w5 = h1_output * d_o
                GRAD_w6 = h2_output * d_o

                w5 += GRAD_w5 * learn_rate
                w6 += GRAD_w6 * learn_rate

                self.o.weights = np.array([w5, w6])

                w1, w3 = self.h1.weights
                w2, w4 = self.h2.weights

                x1_output, x2_output = inputs

                GRAD_w1 = x1_output * d_h1
                GRAD_w2 = x1_output * d_h2
                GRAD_w3 = x2_output * d_h1
                GRAD_w4 = x2_output * d_h2

                w1 += GRAD_w1 * learn_rate
                w2 += GRAD_w2 * learn_rate
                w3 += GRAD_w3 * learn_rate
                w4 += GRAD_w4 * learn_rate

                self.h1.weights = np.array([w1, w3])
                self.h2.weights = np.array([w2, w4])


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
network.train(data, all_y_trues, 1000)

# Make some predictions
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches

# emily = np.array([128, 63]) # 128 pounds, 63 inches
# frank = np.array([155, 68])  # 155 pounds, 68 inches

emily_res = network.feedforward(emily)
frank_res = network.feedforward(frank)


def number_to_sex(number):
    if number >= 0.5:
        return 'F'
    return 'M'


print("Emily: %.3f" % emily_res, number_to_sex(emily_res)) # 0.951 - F
print("Frank: %.3f" % frank_res, number_to_sex(frank_res)) # 0.039 - M