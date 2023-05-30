from fnn import FNN
from data_loader import PKLDataLoader, CSVDataLoader

if __name__ == '__main__':
    training_data, validation_data, test_data = CSVDataLoader.load(r'data/fashion_mnist.zip')
    # training_data, validation_data, test_data = PKLDataLoader.load(r'data/mnist.pkl.gz')
    net = FNN([784, 60, 10], activation='leaky_relu')
    net.fit(training_data, epochs=20, mini_batch_size=10,
            eta=0.001, lambda_=0.01, validation_data=validation_data)
    net.predict(test_data[0], y=test_data[1])
