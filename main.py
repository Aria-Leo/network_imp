import time
from fnn import FNN
from cnn import CNN, Conv2d, Dense, Flatten
from utils.cost import CrossEntropyCost
from data_loader import PKLDataLoader, CSVDataLoader

if __name__ == '__main__':
    training_data, validation_data, test_data = CSVDataLoader.load(r'data/fashion_mnist.zip',
                                                                   visualization=True, samples=500)
    # training_data, validation_data, test_data = PKLDataLoader.load(r'data/mnist.pkl.gz')
    # net = FNN([784, 60, 10], activation='leaky_relu', cost=CrossEntropyCost('softmax'))
    # net.fit(training_data, epochs=10, mini_batch_size=50,
    #         eta=0.001, lambda_=0.01, validation_data=validation_data)
    # net.predict(test_data[0], y=test_data[1])
    net = CNN([
        Conv2d(1, 64, stride=2),
        Conv2d(64, 128, stride=2),
        Conv2d(128, 256, stride=2),
        Conv2d(256, 512, stride=2),
        Flatten(),
        Dense(2 * 2 * 512, 10)
    ])
    time_start = time.time()
    net.fit(training_data[0], training_data[1], epochs=50, batch_size=50,
            validation_data=validation_data)
    net.predict(test_data[0], y=test_data[1])
    time_end = time.time()
    print(f'run cost time: {(time_end - time_start) / 60:.2f}min!')
