# implement logistic regression for ensembling models

from .base import Ensembler
import numpy as np
from sklearn.linear_model import LogisticRegression

class GLMEnsembler(Ensembler):
    def __init__(self) -> None:
        self.model = LogisticRegression()
    
    def _fit(self, x, y):
        self.model.fit(x, y)

    def _predict(self, x):
        return self.model.predict_log_proba(x)