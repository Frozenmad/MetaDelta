from .base import Ensembler
from sklearn.ensemble import RandomForestClassifier

class RFEnsembler(Ensembler):
    def __init__(self) -> None:
        self.model = RandomForestClassifier()
    
    def _fit(self, x, y):
        self.model.fit(x, y)
    
    def _predict(self, x):
        return self.model.predict_log_proba(x)