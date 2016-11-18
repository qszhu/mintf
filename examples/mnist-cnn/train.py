import numpy as np
import tensorflow as tf

from mintf.model import Model
from mintf.layers import Conv2d, MaxPool2x2, Reshape, Dense, Dropout
from mintf.data import DataSource
from mintf.trainer import Trainer

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../mnist_data/', one_hot=True)

# define model
mnist_model = Model()
mnist_model.add(Conv2d(shape=[5, 5, 1, 32], activation=tf.nn.relu))
mnist_model.add(MaxPool2x2())

mnist_model.add(Conv2d(shape=[5, 5, 32, 64], activation=tf.nn.relu))
mnist_model.add(MaxPool2x2())

mnist_model.add(Reshape(shape=[7 * 7 * 64]))
mnist_model.add(Dense(shape=[7 * 7 * 64, 1024], activation=tf.nn.relu))
mnist_model.add(Dropout(0.5))

mnist_model.add(Dense(shape=[1024, 10], activation=tf.nn.softmax, name='output'))

# metrics functions
def loss(y_predict, y_true):
    return -tf.reduce_sum(y_true * tf.log(tf.clip_by_value(y_predict, 1e-10, 1.0)))

def accuracy(y_predict, y_true):
    correct = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_true, 1))
    return tf.reduce_mean(tf.cast(correct, 'float'))

# data source
class MNISTData(DataSource):

    def next_batch(self, n):
        x, y = mnist.train.next_batch(n)
        x = np.reshape(x, (-1, 28, 28, 1))
        return x, y

    def test_data(self):
        x, y = mnist.test.images, mnist.test.labels
        x = np.reshape(x, (-1, 28, 28, 1))
        return x, y

# train
data_src = MNISTData()
trainer = Trainer(mnist_model, [28, 28, 1], [10], loss, accuracy, data_src)

optimizer = tf.train.AdamOptimizer(1e-4)
trainer.train(optimizer, batch_size=50, steps=20000, report_step=200)

# test
x_test, y_test = data_src.test_data()
print 'loss', trainer.loss(x_test, y_test), 'accuracy', trainer.metrics(x_test, y_test)

# export
trainer.export('mnist-cnn.pb', x_test, y_test)
