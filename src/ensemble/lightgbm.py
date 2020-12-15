from .base import Ensembler
import lightgbm as lgb

class GBMEnsembler(Ensembler):
    
    def _fit(self, x, y):
        data = lgb.Dataset(x, label=y)
        self.booster = lgb.train({
            'num_class': max(y) + 1,
            'objective': 'multiclass'
        }, data)
        
    def _predict(self, x):
        return self.booster.predict(x)