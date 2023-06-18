import random
from abc import ABCMeta, abstractmethod
import numpy as np
from collections import deque, defaultdict

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

        self.activation = None
        self.activation_prime = None
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
        print('Flow into dense layer...')
        print(f'Input shape: {inputs.shape}')
        output = inputs @ self.weights.T + self.biases
        self.output_linear__ = output
        if self.activation is not None:
            self.output_activation_prime__ = self.activation_prime(output)
            output = self.activation(output)
        print(f'Output shape: {output.shape}')
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
        print('Flow into conv2d layer...')
        print(f'Input shape: {inputs.shape}')
        if len(inputs.shape) == 3:
            inputs = np.expand_dims(inputs, axis=0)

        # inputs shape: (N, c_out, c_in, h_in, w_in)
        inputs = np.expand_dims(inputs, axis=1).repeat(self.weights.shape[0], axis=1)
        output_linear = Functional.convnd(inputs, self.weights,
                                          stride=self.stride, padding=self.padding)
        output = output_linear.sum(axis=2) + self.biases[np.newaxis, :, np.newaxis, np.newaxis]
        if self.activation is not None:
            self.output_activation_prime__ = self.activation_prime(output)
            output = self.activation(output)
        print(f'Output shape: {output.shape}')

        return output


class Flatten(Layer):

    def __init__(self, start_dim=1, end_dim=-1):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, inputs):
        print('Flow into flatten layer...')
        print(f'Input shape: {inputs.shape}')
        shape_size = len(inputs.shape)
        if self.end_dim < 0:
            end_dim = shape_size + 1 - self.end_dim
        else:
            end_dim = self.end_dim + 1
        assert self.start_dim < end_dim, 'invalid dim input!'
        output = inputs.reshape(*inputs.shape[:self.start_dim], -1, *inputs.shape[end_dim:])
        print(f'Output shape: {output.shape}')
        return output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Adam:

    def __init__(self, eta=0.001, beta1=0.9, beta2=0.999, lambda_=0.01):
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
        current_info['sm'] = (self.beta2 * current_info['sm']
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
        print(f'{self.num_layers}-output layer db, dw, dz calculated')
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
                dz = dz.reshape(last_conv_shape)  # shape: (n, c_last, h_last, w_last)
            else:
                db = np.sum(dz, axis=(0, 2, 3)) / len(labels)  # shape: (c_out,)
                layer_input = self.outputs_[-layer-1]  # shape: (n, c_in, h_in, w_in)
                kernel_size = layer_obj.kernel_size
                padding = layer_obj.padding
                if padding == 'same':
                    padding = kernel_size // 2
                # shape: (c_in, N, h_in, w_in)
                iter_layer_input = layer_input.transpose(1, 0, 2, 3)
                layer_weights = layer_obj.weights  # shape: (c_out, c_in, kernel_size, kernel_size)
                stride = layer_obj.stride
                feature_size = layer_input.shape[-1]
                output_shrink = 1 if feature_size % 2 == 0 else 0
                # shape: (c_out, c_in, N, h_out, w_out)
                iter_dz = dz.transpose(1, 0, 2, 3)
                iter_dz = np.expand_dims(iter_dz, axis=1).repeat(
                    iter_layer_input.shape[0], axis=1)
                dw = Functional.convnd(iter_layer_input, iter_dz, padding=padding,
                                         dilated_kernel=stride - 1,
                                         output_shrink=(0, output_shrink))
                dw = np.sum(dw, axis=2) / len(labels)
                self.gradients['biases'].appendleft(db)
                self.gradients['weights'].appendleft(dw)

                # 计算当前层的dz
                # shape: (c_in, c_out, kernel_size, kernel_size)
                iter_layer_weights = layer_weights.transpose(1, 0, 2, 3)
                layer_input_size = layer_input.shape[-1]
                input_padding = padding
                output_shrink = output_padding = 0
                if input_padding == 0:
                    if layer_input_size % 2 != 0:
                        output_padding = (0, 1)
                else:
                    if layer_input_size % 2 != 0:
                        output_shrink = input_padding
                    else:
                        output_shrink = (input_padding, input_padding - 1)

                # shape: (N, c_in, c_out, h_out, w_out)
                iter_dz = np.expand_dims(dz, axis=1).repeat(
                    iter_layer_weights.shape[0], axis=1)
                dz = Functional.convnd(iter_dz, iter_layer_weights,
                                         padding=kernel_size-1, dilated_feature=stride-1,
                                         conv_mode='math', output_shrink=output_shrink,
                                         output_padding=output_padding)
                dz = np.sum(dz, axis=2) * self.outputs_activation_prime__[-layer-1]
            print(f'{self.num_layers - layer + 1}-{layer_type} layer db, dw, dz calculated')

    def fit(self, features, labels, epochs=10, batch_size=10, validation_data=None):
        # 初始化梯度相关指标的记录
        self.optimizer.init_record()
        # 训练数据总个数
        n = features.shape[0]
        random_index = np.arange(n)
        # 开始训练 循环每一个epochs
        if self.output_dims > 1:
            labels = np.eye(self.output_dims)[labels]
        else:
            labels = labels.reshape(-1, 1)
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
                print('forward complete!')
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
                    encoding_labels = np.eye(self.output_dims)[valid_labels]
                else:
                    encoding_labels = valid_labels.reshape(-1, 1)
                valid_output = self.forward(valid_features)
                total_cost = self.cost.fn(valid_output, encoding_labels) / len(valid_labels)
                print(f'epoch {j} total cost on validation data: {total_cost}')
                if self.output_dims > 1:
                    predict_res = np.argmax(valid_output, axis=1)
                    accuracy = np.sum(valid_labels == predict_res, dtype=int) / len(valid_labels)
                    print(f'epoch {j} accuracy on validation data: {accuracy:.2%}')

        # 清空缓存变量
        self.init_cache()

    def predict(self, X, y=None):
        output = self.forward(X)
        if y is not None:
            if self.output_dims > 1:
                encoding_y = np.eye(self.output_dims)[y]
            else:
                encoding_y = y.reshape(-1, 1)
            total_cost = self.cost.fn(output, encoding_y) / len(y)
            print(f'total cost on test data: {total_cost}')
            if self.output_dims > 1:
                output = np.argmax(output, axis=1)
                accuracy = np.sum(output == y, dtype=int) / len(y)
                print(f'accuracy on test data: {accuracy:.2%}')
        elif self.output_dims > 1:
            output = np.argmax(output, axis=1)

        return output
