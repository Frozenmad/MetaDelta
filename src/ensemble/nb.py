from .base import Ensembler
from sklearn.naive_bayes import GaussianNB
import pickle

class NBEnsembler(Ensembler):
    def __init__(self, model=None) -> None:
        if model is None:
            self.model = GaussianNB()
        else:
            self.model = model

    def _fit(self, x, y):
        self.model.fit(x, y)

    def _predict(self, x):
        return self.model.predict_log_proba(x)

    def serialize(self):
        return pickle.dumps(self.model)
    
    @classmethod
    def construct(cls, content):
        return cls(pickle.loads(content))
