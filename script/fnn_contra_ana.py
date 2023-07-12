import time

import numpy as np
import torch
import torch.utils.data as data_utils

from model.fnn.my_fnn import FNN
from model.fnn.pytorch_fnn import PytorchFNN
from model.fnn.tensorflow_fnn import TensorflowFNN


def my_fnn(training_data, validation_data, test_data):
    net = FNN([784, 60, 10], activation='leaky_relu')
    time_start = time.time()
    history = net.fit(training_data, epochs=50, mini_batch_size=50,
                      eta=0.001, lambda_=0.01, validation_data=validation_data)
    net.predict(test_data[0], y=test_data[1])
    time_end = time.time()
    history.columns = [f'my_fnn_{c}' for c in history.columns]
    print(f'my implement fnn cost time: {(time_end - time_start) / 60:.2f}min!')
    return history


def pytorch_fnn(training_data, validation_data, test_data):
    train_features, train_labels = (torch.tensor(training_data[0], dtype=torch.float32),
                                    torch.from_numpy(training_data[1]))
    train_dataset = data_utils.TensorDataset(train_features, train_labels)
    valid_features, valid_labels = (torch.tensor(validation_data[0], dtype=torch.float32),
                                    torch.from_numpy(validation_data[1]))
    valid_dataset = data_utils.TensorDataset(valid_features, valid_labels)
    test_features, test_labels = (torch.tensor(test_data[0], dtype=torch.float32),
                                  torch.from_numpy(test_data[1]))
    test_dataset = data_utils.TensorDataset(test_features, test_labels)
    net = PytorchFNN()
    time_start = time.time()
    history = net.fit(train_dataset, epochs=50, batch_size=50, validation_dataset=valid_dataset)
    net.predict(test_dataset)
    time_end = time.time()
    history.columns = [f'pytorch_fnn_{c}' for c in history.columns]
    # history = history.loc[:, ~history.columns.str.contains('loss')]
    print(f'pytorch fnn cost time: {(time_end - time_start) / 60:.2f}min!')
    return history


def tensorflow_fnn(training_data, validation_data, test_data):
    train_features, train_labels = training_data[0].astype(np.float32), training_data[1]
    validation_features, validation_labels = validation_data[0].astype(np.float32), validation_data[1]
    test_features, test_labels = test_data[0].astype(np.float32), test_data[1]

    net = TensorflowFNN()
    time_start = time.time()
    history = net.fit(train_features, train_labels, epochs=50, batch_size=50,
                      validation_data=(validation_features, validation_labels))
    test_predict = net.predict(test_features, test_labels)
    print(test_predict.shape)
    time_end = time.time()
    history.columns = [f'tensorflow_fnn_{c}' for c in history.columns]
    # history = history.loc[:, ~history.columns.str.contains('loss')]
    print(f'tensorflow fnn cost time: {(time_end - time_start) / 60:.2f}min!')
    return history
