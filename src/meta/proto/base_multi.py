# base class for prototypical network setup
import torch
from src.utils import timer, logger
from metadl.api.api import MetaLearner
import os
import time

class ProtoMetaLearner(MetaLearner):
    """
    Base model for protonet training
    """
    def __init__(self, meta_epoch, valid_check_epoch, patience, valid_tasks, batch_size, first_eval=1, logger=logger.get_logger('base')) -> None:
        super().__init__()
        self.logger = logger
        self.timer = timer()
        self.timer.initialize(time.time(), 60 * 100)
        self.meta_epoch = meta_epoch
        self.valid_check_epoch = valid_check_epoch
        self.patience = patience
        self.valid_tasks = valid_tasks
        self.batch_size = batch_size
        self.first_eval = first_eval
        self.training_mode = 0
        self.training_stage = 0
        self.saving = False

    def create_model(self, class_num):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError
    
    def meta_fit(self, train_pipe, valid_pipe, cls_num):
        self.timer.begin('preprocess')
        self.class_num = cls_num
        self.create_model(self.class_num)
        time_used = self.timer.end('preprocess')
        self.logger.info('preprocess done, time used', time_used, 'time left', self.timer.time_left())
        epoch_left = None
        p_stage = 0
        pnow = 0
        bestacc = 0.
        exit_condition = 'normal'

        # record current model
        if self.first_eval < 0:
            if self.on_eval_begin(-1):
                acc = 0
                for _ in range(self.valid_tasks):
                    acc += self.eval_one_episode(valid_pipe, mode='valid')
                
                acc /= self.valid_tasks
                _, acc = self.on_eval_end(-1, acc, pnow)
                bestacc = acc
                

                self.logger.info('Beforehand, acc %.6f' % (acc))

        self.saving = True
        self.save_model()
        self.saving = False

        for epoch in range(self.meta_epoch):
            if epoch == self.valid_check_epoch + 1 or (epoch > self.valid_check_epoch and epoch % 100 == 0):
                # run several epochs and estimate time
                max_time = self.timer.query_time_by_name('epoch')
                max_time = (max_time * self.valid_check_epoch + self.timer.query_time_by_name('evaluate', default=50)) / self.valid_check_epoch
                time_left = self.timer.time_left() - self.timer.query_time_by_name('save model') - self.timer.query_time_by_name('evaluate', default=50)
                epoch_left = time_left // max_time + epoch - 1
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

            if epoch >= self.first_eval and epoch % self.valid_check_epoch == 0:
                self.timer.begin('evaluate')
                if self.on_eval_begin(epoch):
                    acc = None
                    for _ in range(self.valid_tasks):
                        if acc is None: acc = self.eval_one_episode(valid_pipe, mode='valid')
                        else: acc += self.eval_one_episode(valid_pipe, mode='valid')

                    acc /= self.valid_tasks
                    do_compare, acc = self.on_eval_end(epoch, acc, pnow)

                    if do_compare:
                        if bestacc < acc:
                            bestacc = acc
                            self.saving = True
                            self.save_model()
                            self.saving = False
                            pnow = 0
                            p_stage = 0
                        self.logger.info('Epoch %3d valid acc %.6f best acc %.6f cur patience %d' % (epoch, acc, bestacc, pnow))
                        pnow += 1
                        p_stage += 1
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
                acc += self.eval_one_episode(valid_pipe, mode='valid')
                self.timer.end('evaluate')
            
            _, acc = self.on_eval_end(-1, acc, -1)
            acc /= self.valid_tasks
            self.logger.info('abnormal exit acc %.6f' % (acc))
            if acc > bestacc:
                self.saving = True
                self.save_model()
                self.saving = False

    def make_learner(self):
        raise NotImplementedError

    def eval_one_episode(self, data_pipe, mode='valid'):
        raise NotImplementedError

    def on_train_begin(self, epoch):
        raise NotImplementedError

    def on_train_end(self, epoch):
        raise NotImplementedError

    def on_eval_begin(self, epoch):
        raise NotImplementedError

    def on_eval_end(self, epoch, acc, patience_now):
        raise NotImplementedError

    def mini_epoch(self, supp, query, epoch, iters):
        raise NotImplementedError

class ProtoMultiManager():
    def __init__(self) -> None:
        self.setup_model()
    
    def setup_model(self):
        raise NotImplementedError

    def fit(self, supp_x, supp_y, img, device='cpu'):
        device = torch.device(device)
        return self.eval_one_episode(supp_x, supp_y, img, device)
    
    def eval_one_episode(self, supp_x, supp_y, img, device):
        raise NotImplementedError

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
        raise NotImplementedError

    def load_model(self, path):
        raise NotImplementedError
