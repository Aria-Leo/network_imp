
import time

import numpy as np
import torch
import torch.utils.data as data_utils

from model.cnn.pytorch_cnn import PytorchCNN
from model.cnn.tensorflow_cnn import TensorflowCNN
from model.cnn.my_cnn import CNN, Conv2d, Dense, Flatten


def my_cnn(training_data, validation_data, test_data):
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


def pytorch_cnn(training_data, validation_data, test_data):
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
    net.predict(test_dataset)
    time_end = time.time()
    print(f'pytorch cnn cost time: {(time_end - time_start) / 60:.2f}min!')
    history.columns = [f'pytorch_cnn_{c}' for c in history.columns]
    # history = history.loc[:, ~history.columns.str.contains('loss')]
    return history


def tensorflow_cnn(training_data, validation_data, test_data):
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
