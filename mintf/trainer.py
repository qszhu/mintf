import tensorflow as tf
import os

class Trainer(object):

    def __init__(self, model, input_shape, output_shape, loss, metrics, data_src, session_cfg=None):
        self._model = model
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._loss_func = loss
        self._metrics_func = metrics
        self._data_src = data_src
        self._session_cfg = session_cfg

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._x = tf.placeholder('float', [None] + input_shape)
            self._y = self._model.create_graph(self._x)

            self._y_ = tf.placeholder('float', [None] + output_shape)
            self._loss = self._loss_func(self._y, self._y_)
            self._metrics = self._metrics_func(self._y, self._y_)

            self._session = tf.Session(config=self._session_cfg)

    def train(self, optimizer, batch_size=100, steps=1000, report_step=100):
        with self._graph.as_default():
            train_step = optimizer.minimize(self._loss)
            self._session.run(tf.initialize_all_variables())

            for i in range(steps):
                batch_x, batch_y = self._data_src.next_batch(batch_size)
                feed_dict = { self._x: batch_x, self._y_: batch_y }
                train_step.run(feed_dict, self._session)
                if i % report_step == 0:
                    train_loss = self._loss.eval(feed_dict, self._session)
                    train_metrics = self._metrics.eval(feed_dict, self._session)
                    print "step %d, training loss %g, metrics %g" % (i, train_loss, train_metrics)

    def metrics(self, x_test, y_test):
        return self._metrics.eval({ self._x: x_test, self._y_: y_test }, self._session)

    def loss(self, x_test, y_test):
        return self._loss.eval({ self._x: x_test, self._y_: y_test }, self._session)

    def export(self, filename, x_test=None, y_test=None):
        g = tf.Graph()
        with g.as_default():
            x = tf.placeholder('float', [None] + self._input_shape, name='input')
            y = self._model.create_graph(x, is_training=False, session=self._session)
            print 'input:', x.name, 'output:', y.name

            sess = tf.Session(config=self._session_cfg)
            sess.run(tf.initialize_all_variables())

            export_dir, export_name = os.path.dirname(filename), os.path.basename(filename)
            tf.train.write_graph(sess.graph.as_graph_def(), export_dir, export_name, as_text=False)

            if x_test != None and y_test != None:
                y_ = tf.placeholder('float', [None] + self._output_shape)
                print self._metrics_func(y, y_).eval({x: x_test, y_: y_test}, sess)

            sess.close()
