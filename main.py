import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.utils.data as data_utils

from pytorch_cnn import PytorchCNN
from tensorflow_cnn import TensorflowCNN
from cnn import CNN, Conv2d, Dense, Flatten
from fnn import FNN
from data_loader import CSVDataLoader

matplotlib.use('TkAgg')
sns.set_style('darkgrid')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def my_dnn_test():
    net = FNN([784, 60, 10], activation='leaky_relu')
    net.fit(training_data, epochs=10, mini_batch_size=50,
            eta=0.001, lambda_=0.01, validation_data=validation_data)
    net.predict(test_data[0], y=test_data[1])


def my_cnn_test():
    net = CNN([
        Conv2d(1, 4, stride=2),
        Conv2d(4, 8, stride=2),
        Conv2d(8, 16, stride=2),
        Conv2d(16, 32, stride=2),
        Conv2d(32, 64, stride=2),
        Flatten(),
        Dense(64, 10)
    ])
    time_start = time.time()
    history = net.fit(training_data[0], training_data[1], epochs=50, batch_size=50,
                      validation_data=validation_data)
    net.predict(test_data[0], y=test_data[1])
    time_end = time.time()
    print(f'my implement cnn cost time: {(time_end - time_start) / 60:.2f}min!')
    history.columns = [f'my_cnn_{c}' for c in history.columns]
    # history = history.loc[:, ~history.columns.str.contains('loss')]
    return history


def pytorch_cnn_test():
    train_features, train_labels = (torch.tensor(training_data[0], dtype=torch.float32),
                                    torch.from_numpy(training_data[1]))
    train_dataset = data_utils.TensorDataset(train_features, train_labels)
    valid_features, valid_labels = (torch.tensor(validation_data[0], dtype=torch.float32),
                                    torch.from_numpy(validation_data[1]))
    valid_dataset = data_utils.TensorDataset(valid_features, valid_labels)
    test_features, test_labels = (torch.tensor(test_data[0], dtype=torch.float32),
                                  torch.from_numpy(test_data[1]))
    test_dataset = data_utils.TensorDataset(test_features, test_labels)
    net = PytorchCNN()
    time_start = time.time()
    history = net.fit(train_dataset, epochs=50, batch_size=50, validation_dataset=valid_dataset)
    history.columns = [f'pytorch_cnn_{c}' for c in history.columns]
    net.predict(test_dataset)
    time_end = time.time()
    print(f'pytorch cnn cost time: {(time_end - time_start) / 60:.2f}min!')
    # history = history.loc[:, ~history.columns.str.contains('loss')]
    return history


def tensorflow_cnn_test():
    train_features, train_labels = training_data[0].astype(np.float32), training_data[1]
    validation_features, validation_labels = validation_data[0].astype(np.float32), validation_data[1]
    test_features, test_labels = test_data[0].astype(np.float32), test_data[1]

    # convert (n, c, h, w) to (n, h, w, c) for tensorflow data format
    train_features = train_features.transpose(0, 2, 3, 1)
    validation_features = validation_features.transpose(0, 2, 3, 1)
    test_features = test_features.transpose(0, 2, 3, 1)
    print(train_features.shape)
    print(validation_features.shape)
    print(test_features.shape)
    net = TensorflowCNN()
    time_start = time.time()
    history = net.fit(train_features, train_labels, epochs=50, batch_size=50,
                      validation_data=(validation_features, validation_labels))
    test_predict = net.predict(test_features, test_labels)
    print(test_predict.shape)
    time_end = time.time()
    print(f'tensorflow cnn cost time: {(time_end - time_start) / 60:.2f}min!')
    history.columns = [f'tensorflow_cnn_{c}' for c in history.columns]
    # history = history.loc[:, ~history.columns.str.contains('loss')]
    return history


if __name__ == '__main__':
    # 使用DNN网络时，需要把CSVDataLoader.load方法的visualization参数设置为False，即保持向量的形式。
    # 使用CNN时，该参数需要设置为True，将输入数据reshape为(n,c,h,w)的图片格式
    training_data, validation_data, test_data = CSVDataLoader.load(r'data/fashion_mnist.zip',
                                                                   visualization=True, training_samples=2000,
                                                                   train_valid_split=0.8, test_samples=500)
    print(f'training features shape: {training_data[0].shape}, labels shape: {training_data[1].shape}')
    print(f'validation features shape: {validation_data[0].shape}, labels shape: {validation_data[1].shape}')
    print(f'test features shape: {test_data[0].shape}, labels shape: {test_data[1].shape}')
    # training_data, validation_data, test_data = PKLDataLoader.load(r'data/mnist.pkl.gz')
    # my_dnn_test()
    my_cnn_history = my_cnn_test()
    pytorch_cnn_history = pytorch_cnn_test()
    tensorflow_cnn_history = tensorflow_cnn_test()
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(211)
    my_cnn_history.plot(y='my_cnn_accuracy', ax=ax)
    pytorch_cnn_history.plot(y='pytorch_cnn_accuracy', ax=ax)
    tensorflow_cnn_history.plot(y='tensorflow_cnn_accuracy', ax=ax)
    ax = fig.add_subplot(212)
    my_cnn_history.plot(y='my_cnn_val_accuracy', ax=ax)
    pytorch_cnn_history.plot(y='pytorch_cnn_val_accuracy', ax=ax)
    tensorflow_cnn_history.plot(y='tensorflow_cnn_val_accuracy', ax=ax)
    plt.show()
