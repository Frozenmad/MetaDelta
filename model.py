import pickle
import time
import random

TIME_LIMIT = 60 * 120 # time limit of the whole process
TIME_TRAIN = TIME_LIMIT - 30 * 120 # set aside 30min for test
t1 = time.time()

import os
import torch

try:
    import numpy as np
except:
    os.system("pip install numpy")

try:
    import cython
except:
    os.system("pip install cython")

try:
    import ot
except:
    os.system("pip install POT")

try:
    import tqdm
except:
    os.system("pip install tqdm")

try:
    import timm
except:
    os.system("pip install timm")

from utils import get_logger, timer, resize_tensor, augment, decode_label, mean
from api import MetaLearner, Learner, Predictor
from backbone import MLP, rn_timm_mix, Wrapper
from torch import optim
import torch.nn.functional as F

# --------------- MANDATORY ---------------
SEED = 98
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)    
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
# -----------------------------------------

LOGGER = get_logger('GLOBAL')
DEVICE = torch.device('cuda')

class MyMetaLearner(MetaLearner):

    def __init__(self, 
                 train_classes: int, 
                 total_classes: int,
                 logger) -> None:

        super().__init__(train_classes, total_classes, logger)
        self.timer = timer()
        self.timer.initialize(time.time(), TIME_TRAIN - time.time() + t1)
        self.timer.begin('load pretrained model')
        self.model = Wrapper(rn_timm_mix(True, 'swsl_resnet50', 0.1)).to(DEVICE)
        
        times = self.timer.end('load pretrained model')
        LOGGER.info('current model', self.model)
        LOGGER.info('load time', times, 's')
        self.dim = 2048

        # only optimize the last 2 layers
        backbone_parameters = []
        backbone_parameters.extend(self.model.set_get_trainable_parameters([3, 4]))
        # set learnable layers
        self.model.set_learnable_layers([3, 4])
        self.cls = MLP(self.dim, train_classes).to(DEVICE)
        self.opt = optim.Adam(
            [
                {"params": backbone_parameters},
                {"params": self.cls.parameters(), "lr": 1e-3}
            ], lr=1e-4
        )

    def meta_fit(self, 
                 meta_train_generator,
                 meta_valid_generator) -> Learner:

        # fix the valid dataset for fair comparison
        valid_task = []
        for task in meta_valid_generator(50):
            # fixed 5-way 5-shot 5-query settings
            supp_x, supp_y = task.support_set[0], task.support_set[1]
            quer_x, quer_y = task.query_set[0], task.query_set[1]
            supp_x = supp_x[supp_y.sort()[1]]
            supp_end = supp_x.size(0)
            valid_task.append([torch.cat([resize_tensor(supp_x, 224), resize_tensor(quer_x, 224)]), quer_y])

        # loop until time runs out
        total_epoch = 0

        # eval ahead
        with torch.no_grad():
            self.model.set_mode(False)
            acc_valid = 0
            for x, quer_y in valid_task:
                x = x.to(DEVICE)
                x = self.model(x)
                supp_x, quer_x = x[:supp_end], x[supp_end:]

                supp_x = supp_x.view(5, 5, supp_x.size(-1))
                logit = decode_label(supp_x, quer_x).cpu().numpy()
                acc_valid += (logit.argmax(1) == np.array(quer_y)).mean()
            acc_valid /= len(valid_task)
            LOGGER.info("epoch %2d valid mean acc %.6f" % (total_epoch, acc_valid))

        best_valid = acc_valid
        best_param = pickle.dumps(self.model.state_dict())

        self.cls.train()
        for i in range(2):
            # train loop
            self.model.set_mode(True)
            for _ in range(5):
                total_epoch += 1
                self.opt.zero_grad()
                err = 0
                acc = 0
                for i, batch in enumerate(meta_train_generator(10)):
                    self.timer.begin('train data loading')
                    X_train, y_train = batch
                    X_train = augment(X_train)
                    X_train = resize_tensor(X_train, 224)
                    X_train = X_train.to(DEVICE)
                    y_train = y_train.view(-1).to(DEVICE)
                    self.timer.end('train data loading')

                    self.timer.begin('train forward')
                    feature = self.model(X_train)
                    logit = self.cls(feature)
                    loss = F.cross_entropy(logit, y_train) / 10.
                    self.timer.end('train forward')

                    self.timer.begin('train backward')
                    loss.backward()
                    self.timer.end('train backward')

                    err += loss.item()
                    acc += logit.argmax(1).eq(y_train).float().mean()

                backbone_parameters = []
                backbone_parameters.extend(self.model.set_get_trainable_parameters([3, 4]))
                torch.nn.utils.clip_grad.clip_grad_norm_(backbone_parameters + list(self.cls.parameters()), max_norm=5.0)
                self.opt.step()
                acc /= 10
                LOGGER.info('epoch %2d error: %.6f acc %.6f | time cost - dataload: %.1f forward: %.1f backward: %.1f' % (
                    total_epoch, err, acc,
                    self.timer.query_time_by_name("train data loading", method=lambda x:mean(x[-10:])),
                    self.timer.query_time_by_name("train forward", method=lambda x:mean(x[-10:])),
                    self.timer.query_time_by_name("train backward", method=lambda x:mean(x[-10:])),
                ))
            
            # eval loop
            with torch.no_grad():
                self.model.set_mode(False)
                acc_valid = 0
                for x, quer_y in valid_task:
                    x = x.to(DEVICE)
                    x = self.model(x)
                    supp_x, quer_x = x[:supp_end], x[supp_end:]

                    supp_x = supp_x.view(5, 5, supp_x.size(-1))
                    logit = decode_label(supp_x, quer_x).cpu().numpy()
                    acc_valid += (logit.argmax(1) == np.array(quer_y)).mean()
                acc_valid /= len(valid_task)
                LOGGER.info("epoch %2d valid mean acc %.6f" % (total_epoch, acc_valid))
            
            if best_valid < acc_valid:
                # save the best model
                best_param = pickle.dumps(self.model.state_dict())

        self.model.load_state_dict(pickle.loads(best_param))
        return MyLearner(self.model.cpu())

class MyLearner(Learner):

    def __init__(self, model=None) -> None:
        super().__init__()
        self.model = model

    @torch.no_grad()
    def fit(self, support_set) -> Predictor:
        self.model.to(DEVICE)
        X_train, y_train, _, n, k = support_set
        X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
        
        return MyPredictor(self.model, X_train, y_train, n, k)

    def save(self, path_to_save: str) -> None:
        torch.save(self.model, os.path.join(path_to_save, "model.pt"))
 
    def load(self, path_to_load: str) -> None:
        if self.model is None:
            self.model = torch.load(os.path.join(path_to_load, 'model.pt'))
    
class MyPredictor(Predictor):

    def __init__(self, model, supp_x, supp_y, n, k) -> None:
        super().__init__()
        self.model = model
        self.other = [supp_x, supp_y, n, k]

    @torch.no_grad()
    def predict(self, query_set: torch.Tensor) -> np.ndarray:
        query_set = query_set.to(DEVICE)
        supp_x, supp_y, n, k = self.other
        supp_x = supp_x[supp_y.sort()[1]]
        end = supp_x.size(0)
        x = self.model(torch.cat([supp_x, query_set]))
        supp_x, quer_x = x[:end], x[end:]
        supp_x = supp_x.view(n, k, supp_x.size(-1))
        return decode_label(supp_x, quer_x).cpu().numpy()
