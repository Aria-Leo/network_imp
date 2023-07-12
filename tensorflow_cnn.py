import pandas as pd
from tensorflow import keras


class CNNModule(keras.Model):

    def __init__(self):
        super(CNNModule, self).__init__()
        self.conv1 = keras.layers.Conv2D(4, kernel_size=3, padding='same',
                                         strides=2, activation='relu')
        self.conv2 = keras.layers.Conv2D(8, kernel_size=3, padding='same',
                                         strides=2, activation='relu',
                                         input_shape=(28, 28, 1))
        self.conv3 = keras.layers.Conv2D(16, kernel_size=3, padding='same',
                                         strides=2, activation='relu',
                                         input_shape=(28, 28, 1))
        self.conv4 = keras.layers.Conv2D(32, kernel_size=3, padding='same',
                                         strides=2, activation='relu',
                                         input_shape=(28, 28, 1))
        self.conv5 = keras.layers.Conv2D(64, kernel_size=3, padding='same',
                                         strides=2, activation='relu',
                                         input_shape=(28, 28, 1))
        self.flatten = keras.layers.Flatten()
        self.output_linear = keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        outputs = self.flatten(outputs)
        outputs = self.output_linear(outputs)
        return outputs


class TensorflowCNN:

    def __init__(self):
        self.model = keras.Sequential()
        self.model.add(CNNModule())

    def fit(self, features, labels, epochs=50, batch_size=50, validation_data=None):
        self.model.build(input_shape=(None, 28, 28, 1))
        print(self.model.summary())
        self.model.compile(loss="sparse_categorical_crossentropy",
                           optimizer=keras.optimizers.experimental.AdamW(
                               learning_rate=0.001, weight_decay=0.01),
                           metrics=["accuracy"])
        history = self.model.fit(features, labels, epochs=epochs, batch_size=batch_size,
                                 validation_data=validation_data)
        return pd.DataFrame(history.history)

    def predict(self, X, y=None):
        predict_y = self.model.predict(X)
        loss, accuracy = self.model.evaluate(X, y)
        print(f'average loss on test data: {loss}')
        print(f'accuracy on test data: {accuracy:.2%}')
        return predict_y.argmax(1)
