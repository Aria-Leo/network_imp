# coding=utf-8
import json
import random

import numpy as np


class Activation:
    @staticmethod
    def sigmoid(z):
        res = 1.0 / (1.0 + np.exp(-z))
        return res

    @staticmethod
    def sigmoid_prime(z):
        return Activation.sigmoid(z) * (1 - Activation.sigmoid(z))

    @staticmethod
    def relu(z):
        return np.where(z > 0, z, 0)

    @staticmethod
    def relu_prime(z):
        return np.where(z > 0, 1, 0)

    @staticmethod
    def leaky_relu(z, decay=0.1):
        return np.where(z > 0, z, decay * z)

    @staticmethod
    def leaky_relu_prime(z, decay=0.1):
        return np.where(z > 0, 1, decay)

    @staticmethod
    def softmax(z):
        """

        :param z: 1-D or 2-D array
        :return:
        """
        z_dim = len(z.shape)
        if z_dim == 1:
            z = z.reshape(1, -1)
        e_z = np.exp(z)
        e_sum = np.sum(e_z, axis=1)
        res = np.divide(e_z.T, e_sum).T
        if z_dim == 1:
            res = res.flatten()
        return res

    @staticmethod
    def softmax_prime(z):
        """
        :param z: 1-D or 2-D array
        :return:
        """
        def base_diff(e):
            """
            :param e: 1-D array
            :return:
            """
            return np.diag(e) - e.reshape(-1, 1) @ e.reshape(1, -1)

        y = Activation.softmax(z)
        z_dim = len(z.shape)
        if z_dim == 1:
            res = base_diff(y)
        else:
            res = np.apply_along_axis(base_diff, 1, y)
        return res


class QuadraticSoftmaxCost:
    @staticmethod
    def fn(a, y, from_logits=False):
        if from_logits:
            a = Activation.softmax(a)
        return 0.5 * np.sum((a - y) ** 2)

    @staticmethod
    def init_delta(z, a, y):
        """

        :param z: 2-D array
        :param a: 2-D array
        :param y: 2-D array
        :return:
        """
        d = a - y
        sd = Activation.softmax_prime(z)
        res = np.array([i_d @ i_sd for i_d, i_sd in zip(d, sd)])
        return res


class CrossEntropySoftmaxCost:
    @staticmethod
    def fn(a, y, from_logits=False):
        if from_logits:
            a = Activation.softmax(a)
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def init_delta(z, a, y):
        d = np.nan_to_num(-y / a + (1 - y) / (1 - a))
        sd = Activation.softmax_prime(z)
        res = np.array([i_d @ i_sd for i_d, i_sd in zip(d, sd)])
        return res


class FNN(object):
    def __init__(self, sizes, activation='relu'):
        # 网络层数
        self.num_layers = len(sizes)
        # 网络每层神经元个数
        self.sizes = sizes
        self.biases = [np.random.randn(y) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.cost = CrossEntropySoftmaxCost
        self.activation = getattr(Activation, activation)
        self.activation_prime = getattr(Activation, f'{activation}_prime')
        self.activation_last = Activation.softmax

        # use for adam
        # fm*: 记录梯度的指数加权平均
        # sm*: 记录梯度平方的指数加权平均
        self.fm_w = None
        self.sm_w = None
        self.fm_b = None
        self.sm_b = None

    def init_adam(self):
        self.fm_w = [np.zeros(w.shape) for w in self.weights]
        self.sm_w = [np.zeros(w.shape) for w in self.weights]
        self.fm_b = [np.zeros(b.shape) for b in self.biases]
        self.sm_b = [np.zeros(b.shape) for b in self.biases]

    def feedforward(self, a):
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = self.activation(a @ w.T + b)
        b, w = self.biases[-1], self.weights[-1]
        a = self.activation_last(a @ w.T + b)
        return a

    def train(self, training_data, epochs=20, mini_batch_size=10, eta=0.001,
              beta1=0.9, beta2=0.999,
              lambda_=0.0, test_data=None):
        # 初始化全局梯度
        self.init_adam()
        train_features, train_labels = training_data
        # 训练数据总个数
        n = train_features.shape[0]
        random_index = np.arange(n)
        # 记录总迭代次数
        iter_counts = 1

        # 开始训练 循环每一个epochs
        for j in range(epochs):
            # 洗牌 打乱训练数据
            random.shuffle(random_index)
            train_features = train_features[random_index]
            train_labels = train_labels[random_index]

            # 训练mini_batch
            for k in range(0, n, mini_batch_size):
                mini_batch = (train_features[k: k + mini_batch_size],
                              train_labels[k: k + mini_batch_size])
                self.adam(mini_batch, iter_counts, eta=eta,
                          beta1=beta1, beta2=beta2, lambda_=lambda_)
                iter_counts += 1

            print("Epoch %s training complete" % j)

            cost = self.total_cost(training_data)
            print("Cost on training data: {}".format(cost))
            accuracy = self.accuracy(training_data, convert=True)
            print("Accuracy on training data: {}".format(accuracy))

            if test_data is not None:
                cost = self.total_cost(test_data, convert=True)
                print("Cost on test data: {}".format(cost))
                accuracy = self.accuracy(test_data)
                print("Accuracy on test data: {}".format(accuracy))

    def adam(self, mini_batch, iter_counts, eta=0.001,
             beta1=0.9, beta2=0.999, lambda_=0.0):
        # 训练每一个mini_batch
        batch_x, batch_y = mini_batch
        nabla_b, nabla_w = self.get_gradient(batch_x, batch_y)
        self.fm_w = [beta1 * fm_w + (1 - beta1) * w for fm_w, w in zip(self.fm_w, nabla_w)]
        self.fm_b = [beta1 * fm_b + (1 - beta1) * b for fm_b, b in zip(self.fm_b, nabla_b)]
        self.sm_w = [beta2 * sm_w + np.nan_to_num((1 - beta2) * w * w) for sm_w, w in zip(self.sm_w, nabla_w)]
        self.sm_b = [beta2 * sm_b + np.nan_to_num((1 - beta2) * b * b) for sm_b, b in zip(self.sm_b, nabla_b)]

        # 初始迭代时放大梯度
        fm_w_rect = [fm_w / (1 - beta1 ** iter_counts) for fm_w in self.fm_w]
        fm_b_rect = [fm_b / (1 - beta1 ** iter_counts) for fm_b in self.fm_b]
        sm_w_rect = [sm_w / (1 - beta2 ** iter_counts) for sm_w in self.sm_w]
        sm_b_rect = [sm_b / (1 - beta2 ** iter_counts) for sm_b in self.sm_b]

        # 更新权重和偏置
        self.weights = [(1 - eta * lambda_) * w - (eta / len(mini_batch)) * fm_w / (np.sqrt(sm_w) + 1e-7)
                        for w, fm_w, sm_w in zip(self.weights, fm_w_rect, sm_w_rect)]
        self.biases = [b - (eta / len(mini_batch)) * fm_b / (np.sqrt(sm_b) + 1e-7)
                       for b, fm_b, sm_b in zip(self.biases, fm_b_rect, sm_b_rect)]

    def get_gradient(self, batch_x, batch_y):
        # 保存每层偏导
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = batch_x

        # 保存每一层的激励值a=activation(z)
        activations = [batch_x]

        # 保存每一层的z=wx+b
        zs = []
        # 前向传播
        layer_count = 1
        for b, w in zip(self.biases, self.weights):
            # 计算每层的z
            z = activation @ w.T + b

            # 保存每层的z
            zs.append(z)

            # 计算每层的y
            if layer_count < self.num_layers - 1:
                activation = self.activation(z)
            else:
                activation = self.activation_last(z)

            # 保存每一层的y
            activations.append(activation)

            layer_count += 1

        # 反向更新
        # 计算最后一层的误差
        delta = self.cost.init_delta(zs[-1], activations[-1], batch_y)

        # 最后一层权重和偏置的导数
        nabla_b[-1] = np.sum(delta, axis=0)
        nabla_w[-1] = delta.T @ activations[-2]

        # 倒数第二层一直到第一层 权重和偏置的导数
        for layer_num in range(2, self.num_layers):
            z = zs[-layer_num]

            sp = self.activation_prime(z)

            # 当前层的误差
            delta = (delta @ self.weights[-layer_num + 1]) * sp

            # 当前层偏置和权重的导数
            nabla_b[-layer_num] = np.sum(delta, axis=0)
            nabla_w[-layer_num] = delta.T @ activations[-layer_num - 1]

        return nabla_b, nabla_w

    def accuracy(self, data, convert=False):
        x, y = data
        a = self.feedforward(x)
        a = np.argmax(a, axis=1)
        if convert:
            y = np.argmax(y, axis=1)
        res = np.sum(a == y, dtype=int) / len(y)
        return f'{res:.2%}'

    def total_cost(self, data, convert=False):
        x, y = data
        a = self.feedforward(x)
        if convert:
            y = np.eye(10)[y]
        cost = self.cost.fn(a, y) / len(y)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "activation": self.activation}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    net = FNN(data['sizes'], activation=data['activation'])
    net.weights = [np.array(w) for w in data['weights']]
    net.biases = [np.array(b) for b in data['biases']]
    return net
