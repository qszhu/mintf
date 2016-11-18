from variables import VariableStore

class Model(object):

    def __init__(self):
        self._vs = VariableStore()
        self._layers = []

    def add(self, layer):
        self._layers.append(layer)

    def create_graph(self, x, is_training=True, session=None):
        if is_training:
            vs = self._vs
        else:
            vs = self._vs.export_variables(session)
        for layer in self._layers:
            x = layer.apply(x, vs, is_training)
        return x
