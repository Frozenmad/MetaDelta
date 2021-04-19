# base class for prototypical network setup
import torch
from src.utils import timer, logger
from metadl.api.api import MetaLearner
import os
import time
from src.utils import DataArgumentor

class ProtoMetaLearner(MetaLearner):
    """
    Base model for protonet training

    You can specify:
    - backbone
    - decoder way
    - loss
    - optimizer
    """
    def __init__(self, meta_epoch, valid_check_epoch, patience, valid_tasks, batch_size, first_eval=1, logger=logger.get_logger('base')) -> None:
        super().__init__()
        self.logger = logger
        self.timer = timer()
        self.timer.initialize(time.time(), 60 * 1000)
        self.meta_epoch = meta_epoch
        self.valid_check_epoch = valid_check_epoch
        self.patience = patience
        self.valid_tasks = valid_tasks
        self.batch_size = batch_size
        self.first_eval = first_eval
        self.data_augmentor = DataArgumentor() if self.use_data_augmentation else None
        self.turn_on_data_augmentor = False

    def create_model(self, class_num):
        pass

    def save_model(self):
        pass
    
    def meta_fit(self, train_pipe, valid_pipe, cls_num):
        self.timer.begin('preprocess')
        self.class_num = cls_num
        self.create_model(self.class_num)
        time_used = self.timer.end('preprocess')
        self.logger.info('preprocess done, time used', time_used, 'time left', self.timer.time_left())
        epoch_left = None
        pnow = 0
        bestacc = 0.
        exit_condition = 'normal'
        self.save_model()
        for epoch in range(self.meta_epoch):
            if epoch == self.valid_check_epoch + 1 or (epoch > self.valid_check_epoch and epoch % 100 == 0):
                # run several epochs and estimate time
                max_time = self.timer.query_time_by_name('epoch')
                max_time = (max_time * self.valid_check_epoch + self.timer.query_time_by_name('evaluate', default=50)) / self.valid_check_epoch
                time_left = self.timer.time_left() - self.timer.query_time_by_name('save model') - self.timer.query_time_by_name('evaluate', default=50)
                epoch_left = time_left // max_time - 10 + epoch
                self.logger.info('=========== estimate time for remaining usage ===========')
                self.logger.info('epoch mean time:', max_time, 'estimate max epoch', epoch_left)
                self.logger.info('eval mean time:', self.timer.query_time_by_name('evaluate', default=50))
                self.logger.info('=========================================================')

            if epoch_left is not None and epoch >= epoch_left:
                self.logger.info('End because of time limit')
                exit_condition = 'time_limit'
                break
            self.timer.begin('epoch')
            if self.on_train_begin(epoch):
                for iters in range(self.batch_size):
                    self.mini_epoch(train_pipe, epoch, iters)
                
            self.on_train_end(epoch)
            self.timer.end('epoch')

            if epoch > self.first_eval and epoch % self.valid_check_epoch == 0:
                self.timer.begin('evaluate')
                if self.on_eval_begin(epoch):
                    acc = 0
                    for _ in range(self.valid_tasks):
                        acc += self.eval_one_episode(valid_pipe)
                        
                    acc /= self.valid_tasks
                    # LOGGER.info('epoch %2d valid acc %.6f' % (epoch, acc))
                    do_compare = self.on_eval_end(epoch, acc, pnow)
                    
                    if do_compare:
                        if bestacc < acc:
                            bestacc = acc
                            self.save_model()
                            pnow = 0
                        self.logger.info('Epoch %3d valid acc %.6f best acc %.6f cur patience %d' % (epoch, acc, bestacc, pnow))
                        pnow += 1
                        if pnow > self.patience:
                            self.logger.info('End because of early stopping')
                            exit_condition = 'early stop'
                            break
                self.timer.end('evaluate')

        if exit_condition == 'time_limit':
            self.on_eval_begin(-1)
            acc = 0
            for _ in range(self.valid_tasks):
                self.timer.begin('evaluate')
                acc += self.eval_one_episode(valid_pipe)
                self.timer.end('evaluate')
            
            self.on_eval_end(-1, acc, -1)
            acc /= self.valid_tasks
            if acc > bestacc:
                self.save_model()

    def eval_one_episode(self, supp, query):
        pass

    def on_train_begin(self, epoch):
        return True

    def on_train_end(self, epoch):
        pass

    def on_eval_begin(self, epoch):
        return True

    def on_eval_end(self, epoch, acc, patience_now):
        return True

    def mini_epoch(self, supp, query, epoch, iters):
        pass

class ProtoMultiManager():
    def __init__(self) -> None:
        self.setup_model()
    
    def setup_model(self):
        pass

    def fit(self, supp_x, supp_y, img, device='cuda:0'):
        device = torch.device(device)
        return self.eval_one_episode(supp_x, supp_y, img, device)
    
    def eval_one_episode(self, supp_x, supp_y, img, device):
        pass

    def save(self, model_dir):
        if(os.path.isdir(model_dir) != True):
            raise ValueError(('The model directory provided is invalid. Please'
                + ' check that its path is valid.'))
        
        # Save a file for the code submission to work correctly.
        self.save_model(model_dir)
            
    def load(self, model_dir):
        if(os.path.isdir(model_dir) != True):
            raise ValueError(('The model directory provided is invalid. Please'
                + ' check that its path is valid.'))
        self.load_model(model_dir)

    def save_model(self, path):
        pass

    def load_model(self, path):
        pass
