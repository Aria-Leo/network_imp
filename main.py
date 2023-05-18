from fnn import FNN
import data_loader

if __name__ == '__main__':
    training_data, validation_data, test_data = data_loader.load_data('data/mnist.pkl.gz')
    net = FNN([784, 60, 10], activation='leaky_relu')
    net.train(training_data, epochs=20, mini_batch_size=10,
              eta=0.001, lambda_=0.01, test_data=test_data)
