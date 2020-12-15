import numpy as np

class Ensembler():
    def __init__(self) -> None:
        pass

    def _fit(self, x, y):
        raise NotImplementedError()

    def _transform_x(self, x):
        return np.array(x).transpose((0, 1)).reshape(len(x[0]), -1)

    def fit(self, x, y):
        return self._fit(self._transform_x(x), y)

    def _predict(self, x):
        raise NotImplementedError()

    def predict(self, x):
        return self._predict(self._transform_x(x))