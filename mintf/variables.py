import tensorflow as tf

class VariableStore(object):

    def __init__(self, variables=None):
        self._variables = variables if variables != None else {}
        self._var_name_counts = {}

    def _get_var_name(self, name):
        cnt = self._var_name_counts.get(name, 0) + 1
        self._var_name_counts[name] = cnt
        return name + str(cnt)

    def _get_variable(self, name, var=None):
        if not name in self._variables and var != None:
            self._variables[name] = var
        return self._variables[name]

    def variable(self, name, init=None, *args, **kw_args):
        if init == None: init = tf.zeros
        var_name = self._get_var_name(name)
        var = tf.Variable(init(*args, **kw_args))
        return self._get_variable(var_name, var)

    def export_variables(self, session):
        res = {}
        for k, v in self._variables.items():
            val = v.eval(session)
            res[k] = tf.constant(val)
        return VariableStore(res)
