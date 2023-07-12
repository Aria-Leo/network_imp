import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from script.cnn_contra_ana import my_cnn, pytorch_cnn, tensorflow_cnn
from script.fnn_contra_ana import my_fnn, pytorch_fnn, tensorflow_fnn
from data_loader import CSVDataLoader

matplotlib.use('TkAgg')
sns.set_style('darkgrid')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def cnn_contra_plot(training_data, validation_data, test_data):
    my_cnn_history = my_cnn(training_data, validation_data, test_data)
    pytorch_cnn_history = pytorch_cnn(training_data, validation_data, test_data)
    tensorflow_cnn_history = tensorflow_cnn(training_data, validation_data, test_data)
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


def fnn_contra_plot(training_data, validation_data, test_data):
    my_fnn_history = my_fnn(training_data, validation_data, test_data)
    pytorch_fnn_history = pytorch_fnn(training_data, validation_data, test_data)
    tensorflow_fnn_history = tensorflow_fnn(training_data, validation_data, test_data)
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(211)
    my_fnn_history.plot(y='my_fnn_accuracy', ax=ax)
    pytorch_fnn_history.plot(y='pytorch_fnn_accuracy', ax=ax)
    tensorflow_fnn_history.plot(y='tensorflow_fnn_accuracy', ax=ax)
    ax = fig.add_subplot(212)
    my_fnn_history.plot(y='my_fnn_val_accuracy', ax=ax)
    pytorch_fnn_history.plot(y='pytorch_fnn_val_accuracy', ax=ax)
    tensorflow_fnn_history.plot(y='tensorflow_fnn_val_accuracy', ax=ax)
    plt.show()


if __name__ == '__main__':
    # 使用DNN时，需要把CSVDataLoader.load方法的visualization参数设置为False，即保持向量的形式。
    # 使用CNN时，该参数需要设置为True，将输入数据reshape为(n,c,h,w)的图片格式
    training_data_cnn, validation_data_cnn, test_data_cnn = CSVDataLoader.load(
        r'data/fashion_mnist.zip', visualization=True, training_samples=2000,
        train_valid_split=0.8, test_samples=500)
    print(f'training features shape: {training_data_cnn[0].shape}, labels shape: {training_data_cnn[1].shape}')
    print(f'validation features shape: {validation_data_cnn[0].shape}, labels shape: {validation_data_cnn[1].shape}')
    print(f'test features shape: {test_data_cnn[0].shape}, labels shape: {test_data_cnn[1].shape}')
    cnn_contra_plot(training_data_cnn, validation_data_cnn, test_data_cnn)

    # training_data_fnn, validation_data_fnn, test_data_fnn = CSVDataLoader.load(
    #     r'data/fashion_mnist.zip', visualization=False, training_samples=None,
    #     train_valid_split=0.8, test_samples=None)
    # print(f'training features shape: {training_data_fnn[0].shape}, labels shape: {training_data_fnn[1].shape}')
    # print(f'validation features shape: {validation_data_fnn[0].shape}, labels shape: {validation_data_fnn[1].shape}')
    # print(f'test features shape: {test_data_fnn[0].shape}, labels shape: {test_data_fnn[1].shape}')
    # fnn_contra_plot(training_data_fnn, validation_data_fnn, test_data_fnn)
