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

    def get_val(self, inputs):
        return np.dot(self.weights, inputs) + self.bias
        # return self.weights[0] * inputs[0] + self.weights[1] * inputs[1] + self.bias

    def feedforward(self, inputs):
        # np.dot - вычисляем скалярное произведение массивов
        return sigmoid(self.get_val(inputs))

    def train(self, inputs, d_L_d_ypred, d_ypred_d_neuron):

        learn_rate = 0.1

        val = self.get_val(inputs)
        # значение без сигмоиды

        deriv_sigmoid_val = deriv_sigmoid(val)

        d_val_d_weight_1 = inputs[0] * deriv_sigmoid_val
        d_val_d_weight_2 = inputs[1] * deriv_sigmoid_val
        # как зависит значение от 1 и 2 весов соответственно

        d_val_d_bias = deriv_sigmoid_val

        tmp = learn_rate * d_L_d_ypred * d_ypred_d_neuron

        weight_1 = self.weights[0] - tmp * d_val_d_weight_1
        weight_2 = self.weights[1] - tmp * d_val_d_weight_2
        self.weights = np.array([weight_1, weight_2])

        self.bias -= tmp * d_val_d_bias

        d_val_d_input_1 = self.weights[0] * deriv_sigmoid_val
        d_val_d_input_2 = self.weights[1] * deriv_sigmoid_val
        return d_val_d_input_1, d_val_d_input_2


class NeuralNetwork:

    def __init__(self):
        # weights = np.array([0, 1])
        # bias = 0

        self.h1 = Neuron()
        self.h2 = Neuron()
        self.o1 = Neuron()

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1

    def train(self, data, y_trues, epochs=1000):
        learn_rate = 0.1

        for epoch in range(epochs):
            for inputs, y_true in zip(data, y_trues):

                h1_val = self.h1.feedforward(inputs)
                h2_val = self.h2.feedforward(inputs)

                o1_inputs = np.array([h1_val, h2_val])
                o1_val = self.o1.feedforward(o1_inputs)
                y_pred = o1_val

                d_L_d_ypred = -2 * (y_true - y_pred)

                d_ypred_d_h1, d_ypred_d_h2 = self.o1.train([h1_val, h2_val], d_L_d_ypred, 1)

                self.h1.train(inputs, d_L_d_ypred, d_ypred_d_h1)
                self.h2.train(inputs, d_L_d_ypred, d_ypred_d_h2)

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