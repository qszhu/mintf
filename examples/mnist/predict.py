import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../mnist_data/', one_hot=True)
X_test, y_test = mnist.test.images, mnist.test.labels

from mintf.predictor import Predictor

predictions = Predictor('mnist.pb').predict(X_test)

def accuracy(y_predict, y_true):
    correct = np.argmax(y_predict, 1) == np.argmax(y_true, 1)
    return np.mean(correct)

print accuracy(predictions, y_test)
