from .base import Ensembler
from sklearn.ensemble import RandomForestClassifier
import pickle

class RFEnsembler(Ensembler):
    def __init__(self, model=None) -> None:
        self.model = model or RandomForestClassifier()
    
    def _fit(self, x, y):
        self.model.fit(x, y)
    
    def _predict(self, x):
        return self.model.predict_log_proba(x)

    def serialize(self):
        return pickle.dumps(self.model)
    
    @classmethod
    def construct(cls, content):
        return cls(pickle.loads(content))
