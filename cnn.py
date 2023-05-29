import random
from abc import ABCMeta, abstractmethod
import numpy as np
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from collections import deque, defaultdict
from functools import partial

from utils.functional import Functional
from utils.cost import CrossEntropyCost


class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward(self, inputs):
        pass


class Dense(Layer):

    def __init__(self, in_features, out_features, activation='softmax'):
        self.in_features = in_features
        self.out_features = out_features
        k = np.sqrt(1 / in_features)
        self.weights = np.random.uniform(-k, k, (out_features, in_features))
        self.biases = np.random.uniform(-k, k, out_features)

        if activation is not None:
            self.activation = getattr(Functional, activation)
            self.activation_prime = getattr(Functional, f'{activation}_prime')

        self.output_linear__ = None
        self.output_activation_prime__ = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputs):
        """
        前向传播
        Args:
            inputs: np.array, shape: (n, in_features)

        Returns:
            output: np.array, shape: (n, out_features)

        """
        output = inputs @ self.weights.T + self.biases
        self.output_linear__ = output
        if self.activation is not None:
            self.output_activation_prime__ = self.activation_prime(output)
            output = self.activation(output)
        return output


class Conv2d(Layer):

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='same', activation='relu'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # initialize
        k = np.sqrt(1 / (in_channels * kernel_size * kernel_size))
        self.weights = np.random.uniform(-k, k, (out_channels, in_channels, kernel_size, kernel_size))
        self.biases = np.random.uniform(-k, k, out_channels)

        self.activation = None
        self.activation_prime = None
        if activation is not None:
            self.activation = getattr(Functional, activation)
            self.activation_prime = getattr(Functional, f'{activation}_prime')

        self.output_activation_prime__ = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputs):
        """
        前向传播
        Args:
            inputs: np.array, shape: (n, c_in, h_in, w_in) or (c_in, h_in, w_in)

        Returns:
            output: np.array, shape: (n, c_out, h_out, w_out)

        """
        if len(inputs.shape) == 3:
            inputs = np.expand_dims(inputs, axis=0)
        with ProcessPoolExecutor(max_workers=3) as executor:
            output_linear = np.array(executor.map(self.__conv_all, inputs))
        output = output_linear.sum(axis=2) + self.biases[np.newaxis, :, np.newaxis, np.newaxis]
        if self.activation is not None:
            self.output_activation_prime__ = self.activation_prime(output)
            output = self.activation(output)

        return output

    def __conv_all(self, single_input):
        """

        Args:
            single_input: np.array, shape: (c_in, h_in, w_in)

        Returns:
            outputs: np.array, shape: (c_out, c_in, h_out, w_out)

        """
        conv_single = partial(Functional.conv2d, single_input, stride=self.stride, padding=self.padding)
        with Pool(processes=3) as pool:
            outputs = np.array(pool.map(conv_single, self.weights))
        return outputs


class Flatten(Layer):

    def __init__(self, start_dim=1, end_dim=-1):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, inputs):
        shape_size = len(inputs.shape)
        if self.end_dim < 0:
            end_dim = shape_size + 1 - self.end_dim
        else:
            end_dim = self.end_dim + 1
        assert self.start_dim < end_dim, 'invalid dim input!'
        output = inputs.reshape(*inputs.shape[:self.start_dim], -1, *inputs.shape[end_dim:])
        return output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Adam:

    def __init__(self, eta=0.001, beta1=0.9, beta2=0.999, lambda_=0.0):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.lambda_ = lambda_

        self.gradient_index_info = None
        self.init_record()

    def init_record(self):
        self.gradient_index_info = defaultdict(dict)

    def update(self, name, parameter, parameter_gradient):
        # 初始化累积梯度
        if name not in self.gradient_index_info:
            # 存储梯度的指数加权平均
            self.gradient_index_info[name]['fm'] = np.zeros(parameter.shape)
            # 存储梯度平方的指数加权平均
            self.gradient_index_info[name]['sm'] = np.zeros(parameter.shape)
            # 存储迭代次数
            self.gradient_index_info[name]['iter_counts'] = 1
        else:
            self.gradient_index_info[name]['iter_counts'] += 1

        current_info = self.gradient_index_info[name]
        current_info['fm'] = self.beta1 * current_info['fm'] + (1 - self.beta1) * parameter_gradient
        current_info['sm'] = (self.beta2 * current_info['fm']
                              + np.nan_to_num((1 - self.beta2) * parameter_gradient * parameter_gradient))

        # 在初始迭代时需要放大梯度
        fm_rect = current_info['fm'] / (1 - self.beta1 ** current_info['iter_counts'])
        sm_rect = current_info['sm'] / (1 - self.beta2 ** current_info['iter_counts'])

        # 更新参数
        new_parameter = ((1 - self.eta * self.lambda_) * parameter
                         - self.eta * fm_rect / (np.sqrt(sm_rect) + 1e-7))
        return new_parameter


class CNN:

    def __init__(self, sequential: list, cost=CrossEntropyCost(), optimizer=Adam()):
        """

        Args:
            sequential: list,
                example: [
                    Conv2d(3, 256, 3, 2),
                    Conv2d(256, 512, 3, 2),
                    ...
                    Flatten(),
                    Dense(1024, 10)
                ]
                注意，网络结构只能前面是卷积层，经过Flatten层后为全连接层
        """
        self.sequential = sequential
        self.num_layers = len(sequential) + 1  # 输入也看作一层
        # 大于1表示分类问题，等于1表示回归问题
        self.output_dims = sequential[-1].out_features

        # 依赖注入
        self.cost = cost
        self.optimizer = optimizer

        self.outputs_ = []
        self.outputs_activation_prime__ = []
        self.outputs_linear_last__ = None
        # 保存每层的梯度
        self.gradients = defaultdict(deque)

    def init_cache(self):
        self.outputs_ = []
        self.outputs_activation_prime__ = []
        self.outputs_linear_last__ = None
        # 保存每层的梯度
        self.gradients = defaultdict(deque)

    def forward(self, features, record=False):
        output = features
        if record:
            self.outputs_.append(output)
            self.outputs_activation_prime__.append(1)
        for obj in self.sequential:
            output = obj(output)
            if record:
                layer_type = obj.__class__.__name__
                self.outputs_.append(output)
                if layer_type != 'Flatten':
                    self.outputs_activation_prime__.append(obj.output_activation_prime__)
                else:
                    pre_prime = self.outputs_activation_prime__[-1]
                    current_prime = obj(pre_prime)
                    self.outputs_activation_prime__.append(current_prime)
        if record:
            self.outputs_linear_last__ = self.sequential[-1].output_linear__
        return output

    def backward(self, labels):
        """
        calculate gradients per layer
        Returns:

        """
        # 首先计算最后一层的梯度
        delta = self.cost.init_delta(self.outputs_linear_last__, self.outputs_[-1], labels)
        db = np.sum(delta, axis=0) / len(labels)
        dw = (delta.T @ self.outputs_[-2]) / len(labels)
        self.gradients['biases'].appendleft(db)
        self.gradients['weights'].appendleft(dw)
        # 获取loss对最后一层输入的导数, shape: (n, features)
        dz = (delta @ self.sequential[-1].weights) * self.outputs_activation_prime__[-2]
        for layer in range(2, self.num_layers):
            layer_obj = self.sequential[-layer]
            layer_type = layer_obj.__class__.__name__
            if layer_type == 'Dense':
                db = np.sum(dz, axis=0) / len(labels)
                dw = (dz.T @ self.outputs_[-layer-1]) / len(labels)
                self.gradients['biases'].appendleft(db)
                self.gradients['weights'].appendleft(dw)
                dz = (dz @ self.sequential[-layer]) * self.outputs_activation_prime__[-layer-1]
            elif layer_type == 'Flatten':
                self.gradients['biases'].appendleft(None)
                self.gradients['weights'].appendleft(None)
                last_conv_shape = self.outputs_[-layer-1].shape
                dz = dz.reshape(*last_conv_shape)  # shape: (n, c_last, h_last, w_last)
            else:
                db = np.sum(dz, axis=(0, 2, 3)) / len(labels) # shape: (c_out,)
                layer_input = self.outputs_[-layer-1]  # shape: (n, c_in, h_in, w_in)
                padding = layer_obj.padding
                stride = layer_obj.stride
                kernel_size = layer_obj.kernel_size
                iter_dz = dz.transpose(1, 0, 2, 3)
                iter_layer_input = layer_input.transpose(1, 0, 2, 3)
                dw = []  # desired shape: (c_out, c_in, kernel_size, kernel_size)
                layer_weights = layer_obj.weights  # shape: (c_out, c_in, kernel_size, kernel_size)
                for sub_dz in iter_dz:
                    for sub_layer_input in iter_layer_input:
                        # sub_dw shape: (n, kernel_size, kernel_size)
                        sub_dw = Functional.conv2d(sub_layer_input, sub_dz, padding=padding, dilated_kernel=stride-1)
                        sub_dw = np.sum(sub_dw, axis=0)
                        dw.append(sub_dw)
                dw = np.reshape(dw, *layer_weights.shape) / len(labels)
                self.gradients['biases'].appendleft(db)
                self.gradients['weights'].appendleft(dw)
                iter_layer_weights = layer_weights.transpose(1, 0, 2, 3)
                # 计算当前层的dz, shape: (n, c_in, h_in, w_in)
                current_dz = []
                for sub_dz in dz:
                    for sub_layer_weights in iter_layer_weights:
                        # 求解dz是数学中的卷积
                        sub_current_dz = Functional.conv2d(sub_dz, sub_layer_weights,
                                                           padding=kernel_size-1, dilated_feature=stride-1,
                                                           conv_mode='math')
                        sub_current_dz = np.sum(sub_current_dz, axis=0)
                        current_dz.append(sub_current_dz)
                dz = np.reshape(current_dz, *layer_input.shape) * self.outputs_activation_prime__[-layer-1]

    def fit(self, features, labels, epochs=10, batch_size=10, validation_data=None):
        # 初始化梯度相关指标的记录
        self.optimizer.init_record()
        # 训练数据总个数
        n = features.shape[0]
        random_index = np.arange(n)
        # 开始训练 循环每一个epochs
        for j in range(epochs):
            # 洗牌 打乱训练数据
            random.shuffle(random_index)
            features = features[random_index]
            labels = labels[random_index]

            # 训练mini_batch
            total_cost = 0
            accuracy = 0
            for k in range(0, n, batch_size):
                # 初始化缓存
                self.init_cache()
                batch_features = features[k: k + batch_size]
                batch_labels = labels[k: k + batch_size]
                # 前向传播
                batch_output = self.forward(batch_features, record=True)
                # 反向梯度推导
                self.backward(batch_labels)
                # 根据梯度更新参数
                layer = 1
                for weight_gradient, bias_gradient in zip(self.gradients['weights'], self.gradients['biases']):
                    if weight_gradient is None or bias_gradient is None:
                        layer += 1
                        continue
                    weight = self.sequential[layer-1].weights
                    bias = self.sequential[layer-1].biases
                    layer_weight_name = f'{layer}_weight'
                    layer_bias_name = f'{layer}_bias'
                    self.sequential[layer-1].weights = self.optimizer.update(
                        layer_weight_name, weight, weight_gradient)
                    self.sequential[layer-1].biases = self.optimizer.update(
                        layer_bias_name, bias, bias_gradient)
                    layer += 1

                total_cost += self.cost.fn(batch_output, batch_labels)
                if self.output_dims > 1:
                    true_res = np.argmax(batch_labels, axis=1)
                    predict_res = np.argmax(batch_output, axis=1)
                    accuracy += np.sum(true_res == predict_res, dtype=int)

            # 计算总平均损失
            total_cost /= n
            print(f'epoch {j} total cost on training data: {total_cost}')
            if self.output_dims > 1:
                # 计算准确率
                accuracy /= n
                print(f'epoch {j} accuracy on training data: {accuracy:.2%}')

            if validation_data is not None:
                valid_features, valid_labels = validation_data
                if self.output_dims > 1:
                    valid_labels = np.eye(self.output_dims)[valid_labels]
                else:
                    valid_labels = valid_labels.reshape(-1, 1)
                valid_output = self.forward(valid_features)
                total_cost = self.cost.fn(valid_output, valid_labels) / len(valid_labels)
                print(f'epoch {j} total cost on validation data: {total_cost}')
                if self.output_dims > 1:
                    true_res = np.argmax(valid_labels, axis=1)
                    predict_res = np.argmax(valid_output, axis=1)
                    accuracy = np.sum(true_res == predict_res, dtype=int) / len(valid_labels)
                    print(f'epoch {j} accuracy on validation data: {accuracy:.2%}')

        # 清空缓存变量
        self.init_cache()

    def predict(self, features):
        output = self.forward(features)
        return output
