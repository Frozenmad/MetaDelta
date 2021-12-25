# implement logistic regression for ensembling models

import pickle
from .base import Ensembler
import numpy as np
from sklearn.linear_model import LogisticRegression

class GLMEnsembler(Ensembler):
    def __init__(self, model=None) -> None:
        if model is None:
            self.model = LogisticRegression()
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
