# coding=utf-8
import json
import random

import numpy as np

from utils.functional import Functional
from utils.cost import CrossEntropyCost


class FNN:
    def __init__(self, sizes, cost=CrossEntropyCost(), activation='relu'):
        # 网络层数
        self.num_layers = len(sizes)
        # 网络每层神经元个数
        self.sizes = sizes
        self.output_dims = self.sizes[-1]
        self.biases = [np.random.randn(y) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.cost = cost
        self.activation = getattr(Functional, activation)
        self.activation_prime = getattr(Functional, f'{activation}_prime')

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
        a = a @ w.T + b
        if self.cost.activation is not None:
            a = self.cost.activation(a)
        return a

    def fit(self, training_data, epochs=20, mini_batch_size=10, eta=0.001,
            beta1=0.9, beta2=0.999,
            lambda_=0.0, validation_data=None):
        """
        model training
        Args:
            training_data: tuple->(features, labels),
                features shape: (n_samples, n_features),
                labels shape: (n_samples, n_classes)(one-hot encoding or other encoding format)
            epochs:
            mini_batch_size:
            eta: learning rate
            beta1: corresponding to adam beta1
            beta2: corresponding to adam beta2
            lambda_: the regularization of weights
            validation_data: tuple->(features, labels),
                features shape: (n_samples, n_features),
                labels shape: (n_samples, )

        Returns:

        """
        # 初始化全局梯度
        self.init_adam()
        train_features, train_labels = training_data
        if self.output_dims > 1:
            train_labels = np.eye(self.output_dims)[train_labels]
        else:
            train_labels = train_labels.reshape(-1, 1)
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

            print(f'Epoch {j} training complete')

            cost = self.total_cost(training_data, convert=True)
            print(f'Cost on training data: {cost}')
            if self.output_dims > 1:
                accuracy = self.accuracy(training_data)
                print(f'Accuracy on training data: {accuracy}')

            if validation_data is not None:
                cost = self.total_cost(validation_data, convert=True)
                print(f'Cost on validation data: {cost}')
                if self.output_dims > 1:
                    accuracy = self.accuracy(validation_data)
                    print(f'Accuracy on validation data: {accuracy}')

    def predict(self, X, y=None):
        output_y = self.feedforward(X)
        if y is not None:
            cost = self.total_cost((X, y), predict=output_y, convert=True)
            print(f'Cost on test data: {cost}')
            if self.output_dims > 1:
                accuracy = self.accuracy((X, y), predict=output_y)
                print(f'Accuracy on test data: {accuracy}')
                output_y = np.argmax(output_y, axis=1)
        return output_y

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
            elif self.cost.activation is not None:
                activation = self.cost.activation(z)

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

    def accuracy(self, data, predict=None, encoding=False):
        """
        计算准确率，只有分类问题才可以调用该方法
        Args:
            data:
            predict: data对应的预测结果
            encoding: 为True时，表示每个样本的标签为编码向量

        Returns:

        """
        x, y = data
        if predict is None:
            a = self.feedforward(x)
        else:
            a = predict
        a = np.argmax(a, axis=1)
        if encoding:
            y = np.argmax(y, axis=1)
        res = np.sum(a == y, dtype=int) / len(y)
        return f'{res:.2%}'

    def total_cost(self, data, predict=None, convert=False):
        """
        计算损失
        Args:
            data:
            predict: data对应的预测结果
            convert: 分类问题才使用到该参数；等于True时，说明每个样本的标签为标量，需要将标签转换为编码向量
                注：这里由于我对训练数据标签的编码方式为one-hot编码，就暂时写死了，读者可以对编码方式进行替换

        Returns:

        """
        x, y = data
        if predict is None:
            a = self.feedforward(x)
        else:
            a = predict
        if self.output_dims > 1 and convert:  # 分类问题
            y = np.eye(self.output_dims)[y]
        elif self.output_dims == 1:  # 回归问题
            y = y.flatten()
            a = a.flatten()
        cost = self.cost.fn(a, y) / len(y)
        return cost

    def save(self, filename):
        data = {'sizes': self.sizes,
                'weights': [w.tolist() for w in self.weights],
                'biases': [b.tolist() for b in self.biases],
                'cost': self.cost,
                'activation': self.activation}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    net = FNN(data['sizes'], cost=data['cost'], activation=data['activation'])
    net.weights = [np.array(w) for w in data['weights']]
    net.biases = [np.array(b) for b in data['biases']]
    return net
