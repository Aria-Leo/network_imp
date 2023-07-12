import pandas as pd
from tensorflow import keras


class FNNModule(keras.Model):

    def __init__(self):
        super(FNNModule, self).__init__()
        self.dense = keras.layers.Dense(60, activation='leaky_relu')
        self.output_linear = keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        outputs = self.dense(inputs)
        outputs = self.output_linear(outputs)
        return outputs


class TensorflowFNN:

    def __init__(self):
        self.model = keras.Sequential()
        self.model.add(FNNModule())

    def fit(self, features, labels, epochs=50, batch_size=50, validation_data=None):
        self.model.build(input_shape=(None, 784))
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
