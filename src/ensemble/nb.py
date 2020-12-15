from .base import Ensembler
from sklearn.naive_bayes import GaussianNB

class NBEnsembler(Ensembler):
    def __init__(self) -> None:
        self.model = GaussianNB()

    def _fit(self, x, y):
        self.model.fit(x, y)

    def _predict(self, x):
        return self.model.predict_log_proba(x)