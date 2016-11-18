import tensorflow as tf

class Layer(object):

    def __init__(self, shape=None, activation=None, name=None):
        self._shape = shape
        self._activation = activation if activation != None else tf.identity
        self._name = name

    def apply(self, x):
        return self._activation(x, name=self._name)

class Reshape(Layer):

    def apply(self, x, vs=None, is_training=True):
        y = tf.reshape(x, [-1] + self._shape)
        return super(Reshape, self).apply(y)

class Dense(Layer):

    def apply(self, x, vs=None, is_training=True):
        W = vs.variable('weight', tf.truncated_normal, self._shape, stddev=1e-3)
        b = vs.variable('bias', tf.constant, 1e-3, shape=[self._shape[-1]])
        y = tf.add(tf.matmul(x, W), b)
        return super(Dense, self).apply(y)

class Conv2d(Layer):

    def apply(self, x, vs=None, is_training=True):
        W = vs.variable('weight', tf.truncated_normal, self._shape, stddev=1e-3)
        b = vs.variable('bias', tf.constant, 1e-3, shape=[self._shape[-1]])
        y = tf.add(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'), b)
        return super(Conv2d, self).apply(y)

class MaxPool2x2(Layer):

    def apply(self, x, vs=None, is_training=True):
        y = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return super(MaxPool2x2, self).apply(y)

class Dropout(Layer):

    def __init__(self, keep_prob, shape=None, activation=None, name=None):
        super(Dropout, self).__init__(shape, activation, name)
        self._keep_prob = keep_prob

    def apply(self, x, vs=None, is_training=True):
        if is_training:
            y = tf.nn.dropout(x, self._keep_prob)
            return super(Dropout, self).apply(y)
        return super(Dropout, self).apply(x)

class Spread(Layer):

    def __init__(self, layer, spread, shape=None, activation=None, name=None):
        super(Spread, self).__init__(shape, activation, name)
        self._layer = layer
        self._spread = spread

    def apply(self, x, vs=None, is_training=True):
        y = tf.pack([self._layer.apply(x, vs, is_training) for i in range(self._spread)], axis=1)
        return super(Spread, self).apply(y)
