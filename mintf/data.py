class DataSource(object):

    def next_batch(self, n):
        raise NotImplementedError

    def test_data(self):
        raise NotImplementedError
