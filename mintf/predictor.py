import tensorflow as tf

class Predictor(object):

    def __init__(self, filename, input_name='input:0', output_name='output:0'):
        self._filename = filename
        self._input_name = input_name
        self._output_name = output_name

    def predict(self, input_data):
        g = tf.Graph()
        with g.as_default():
            with tf.gfile.FastGFile(self._filename, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')

            with tf.Session() as sess:
                output_tensor = sess.graph.get_tensor_by_name(self._output_name)
                predictions = sess.run(output_tensor, { self._input_name: input_data })
                return predictions
