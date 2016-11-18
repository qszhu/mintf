import tensorflow as tf

from mintf.model import Model
from mintf.layers import Dense
from mintf.data import DataSource
from mintf.trainer import Trainer

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../mnist_data/', one_hot=True)

# define model
mnist_model = Model()
mnist_model.add(Dense([784, 10], activation=tf.nn.softmax, name='output'))

# metrics functions
def loss(y_predict, y_true):
    return -tf.reduce_sum(y_true * tf.log(tf.clip_by_value(y_predict, 1e-10, 1.0)))

def accuracy(y_predict, y_true):
    correct = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_true, 1))
    return tf.reduce_mean(tf.cast(correct, 'float'))

# data source
class MNISTData(DataSource):

    def next_batch(self, n):
        return mnist.train.next_batch(n)

    def test_data(self):
        return mnist.test.images, mnist.test.labels

# train
data_src = MNISTData()
trainer = Trainer(mnist_model, [784], [10], loss, accuracy, data_src)

optimizer = tf.train.GradientDescentOptimizer(0.01)
trainer.train(optimizer)

# test
x_test, y_test = data_src.test_data()
print 'loss', trainer.loss(x_test, y_test), 'accuracy', trainer.metrics(x_test, y_test)

# export
trainer.export('mnist.pb', x_test, y_test)
