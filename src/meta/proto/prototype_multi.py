import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_multi import ProtoMetaLearner, ProtoMultiManager
from .label_propagation import map_label_propagation
from ...learner.decoder.classifier import MLP
from ...learner.pretrained_encoders import enet_mixup, rn_timm_mix, Wrapper
from ...learner.pretrained_encoders.efficientnet import enet_mixup
from ...learner.pretrained_encoders.resnet import rn_timm_mix
from src.utils import logger
from torchvision import transforms

TRAIN_AUGMENT = transforms.Compose([
    transforms.Normalize(-1.0, 2.0/255.0),
    transforms.RandomCrop(128, padding=16),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(127.5,127.5)
])

def normalize(emb):
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb

def resize_tensor(x,size):
    return transforms.functional.resize(x, [size, size], transforms.functional.InterpolationMode.BILINEAR, antialias=True)

def augment(x):
    return TRAIN_AUGMENT(x)
    #return x

def mean(x):
    return sum(x) / len(x)

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def process_data(supp, query, train=True, config=None):
    with torch.no_grad():
        if train:

            tmp_support_x = augment(supp[0])
            tmp_query_x = augment(query[0])
            way = config['way']
            others = supp[0].size()[1:]

            _, supp_slices = supp[1].sort()
            supp_x = tmp_support_x[supp_slices].view(way, config['episode_train_shot'], *others)
            supp_y = supp[2][supp_slices].view(way, config['episode_train_shot'])

            if config['pretrain_shot'] <= config['episode_train_shot']:
                x = supp_x[:, :config['pretrain_shot']].reshape(way * config['pretrain_shot'], *others)
                y = supp_y[:, :config['pretrain_shot']].reshape(-1)
            else:
                # combine supp and query
                more = config['pretrain_shot'] - config['episode_train_shot']
                _, quer_slices = query[1].sort()
                quer_x = tmp_query_x[quer_slices].reshape(way, config['episode_train_query'], *others)[:, :more]
                quer_y = query[2][quer_slices].reshape(way, config['episode_train_query'])[:, :more]
                x = torch.cat([supp_x, quer_x], dim=1).reshape(way * config['pretrain_shot'], *others)
                y = torch.cat([supp_y, quer_y], dim=1).reshape(-1)
            
            return [x, y]

        else:
            # load valid data
            return [supp, query]

MODEL = {
    'enet': lambda *args, **kwargs: Wrapper(enet_mixup(*args, **kwargs)),
    'rn': lambda *args, **kwargs: Wrapper(rn_timm_mix(*args, **kwargs))
}

def whiten(features):
    if len(features.shape) == 3:
        w, s, d = features.shape
        features_2d = features.view(w * s, d)
    else:
        features_2d = features
    features_2d = features_2d - features_2d.mean(dim=0, keepdim=True)
    features_2d = normalize(features_2d)
    if len(features.shape) == 3:
        return features_2d.view(w, s, d)
    return features_2d

def decode_label(sx, qx, prob=True):
    sx = whiten(sx)
    qx = whiten(qx)

    lg = map_label_propagation(qx, sx)

    if not prob:
        lg = torch.log(lg + 1e-6)

    return lg

class MyMetaLearner(ProtoMetaLearner):
    def __init__(self, config=None) -> None:
        self.__dict__.update(config)
        self.config = config
        self.logger = logger.get_logger('proto_{}'.format(self.global_id))
        
        super().__init__(self.epochs, self.eval_epoch, self.patience, self.eval_tasks, self.batch_size, self.first_eval, self.logger)
        self.device = torch.device(self.device)
        self.logger.info('current hp', config)
        self.logger_performance = logger.get_logger('proto_{}'.format(self.global_id), 'valid.txt')

    def load_from_best(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'proto_{}'.format(self.global_id))
        path_to_tmp = os.path.join(path, 'tmp.pt')
        m = torch.load(path_to_tmp)
        self.model = m['model'].to(self.device)
        print("loading the best model from before")

    def create_model(self, class_num):
        self.timer.begin('load pretrained model')
        
        if self.backbone == "enet":
            self.model = Wrapper(enet_mixup(True, self.name, self.momentum))
        else:
            self.model = Wrapper(rn_timm_mix(True, self.name, self.momentum))

        self.model.to(self.device)

        # for origin class training
        times = self.timer.end('load pretrained model')
        self.logger.info('current model', self.model)
        self.logger.info('load time', times, 's')

        self.model.set_mode(False)
        self.dim = self.model(torch.randn(2,3,self.size,self.size).to(self.device)).size()[-1]

        self.model.set_mode(True)
        self.logger.info('detect encoder dimension', self.dim)
        
        backbone_parameters = []
        backbone_parameters.extend(self.model.set_get_trainable_parameters(self.parameters))
        # set learnable layers
        self.model.set_learnable_layers(self.parameters)
        self.cls = MLP(self.dim, class_num).to(self.device)
        self.opt = optim.Adam(
            [
                {"params": backbone_parameters},
                {"params": self.cls.parameters(), "lr": self.lr_head}
            ], lr=self.lr_backbone
        )
    
    def on_train_begin(self, epoch):
        self.model.set_mode(True)
        self.err_list = []
        self.acc_list = []
        self.cls.train()
        self.opt.zero_grad()
        return True

    def on_train_end(self, epoch):
        backbone_parameters = []
        backbone_parameters.extend(self.model.set_get_trainable_parameters(self.parameters))

        if self.clip_norm > 0:
            nn.utils.clip_grad.clip_grad_norm_(backbone_parameters + list(self.cls.parameters()), max_norm=self.clip_norm)

        self.opt.step()
        
        err = sum(self.err_list)
        acc = mean(self.acc_list)
        
        self.logger.info('epoch %2d mode: train error %.6f acc %.6f | []: %.1f ->: %.1f <-: %.1f' % (
                epoch, err, acc, 
                self.timer.query_time_by_name("train data loading", method=lambda x:mean(x[-self.batch_size:])),
                self.timer.query_time_by_name("train forward", method=lambda x:mean(x[-self.batch_size:])),
                self.timer.query_time_by_name("train backward", method=lambda x:mean(x[-self.batch_size:])),
            )
        )

    def mini_epoch(self, train_pipe, epoch, iters):
        self.timer.begin('train data loading')
        x, y = train_pipe.recv()
        if x.size(2) != self.size:
            x = resize_tensor(x, self.size)
        train_pipe.send(True)
        self.timer.end('train data loading')
        self.timer.begin('train forward')
        x = x.to(self.device)
        y = y.to(self.device)
        
        feature = self.model(x, film=self.film_model)
        logit = self.cls(feature)
            
        loss = F.cross_entropy(logit, y)
        loss = loss / self.batch_size
        self.timer.end('train forward')
        self.timer.begin('train backward')
        loss.backward()
        self.timer.end('train backward')
        self.err_list.append(loss.item())
        self.acc_list.append(accuracy(logit, y))
    
    @torch.no_grad()
    def eval_one_episode(self, valid_pipe, mode='valid'):
        
        supp, query = valid_pipe.recv()
        if supp[0].size(2) != self.size:
            supp[0] = resize_tensor(supp[0], self.size)
            query[0] = resize_tensor(query[0], self.size)
        valid_pipe.send(True)
        _, slices = supp[1].to(self.device).sort()
        supp_x, quer_x = supp[0].to(self.device)[slices], query[0].to(self.device)
        supp_end = supp_x.size(0)
        x = self.model(torch.cat([supp_x, quer_x], dim=0))
        supp_x, quer_x = x[:supp_end], x[supp_end:]

        supp_x = supp_x.view(5, -1, supp_x.size(-1))
        logit = decode_label(supp_x, quer_x)
        if isinstance(logit, torch.Tensor):
            logit = np.array(logit.detach().cpu())
        acc = (logit.argmax(1) == np.array(query[1])).mean()
        return acc

    def on_eval_begin(self, epoch):
        self.model.set_mode(False)
        return True
    
    def on_eval_end(self, epoch, acc, patience_now):
        self.logger_performance.info(' %3d: %.5f' % (epoch, acc))
        return True, acc

    def save_model(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'proto_{}'.format(self.global_id))
        os.makedirs(path, exist_ok=True)
        torch.save(self.model, os.path.join(path, 'tmp.pt'))

class MyMultiManager(ProtoMultiManager):
    def __init__(self, model=None, config=None) -> None:
        self.model = model
        self.config = config
        self.loaded = False
        self.resizer = None

    def load_model(self, path):
        if self.loaded:
            return
        self.model = torch.load(os.path.join(path, 'model.pt'))
        self.config = torch.load(os.path.join(path, 'config.pt'))
        self.loaded = True

    def save_model(self, path):
        torch.save(self.model, os.path.join(path, 'model.pt'))
        torch.save(self.config, os.path.join(path, 'config.pt'))
    
    def to(self, device):
        self.model.to(device)

    def eval_one_episode(self, supp_x, supp_y, img, device):
        self.model.to(device)
        
        # resize to the shape of input
        if not (supp_x.size(2) == supp_x.size(3) == self.config['size']):
            supp_x = resize_tensor(supp_x,self.config['size'])
            img = resize_tensor(img,self.config['size'])

        supp_x = supp_x.to(device)
        supp_y = supp_y.to(device)
        img = img.to(device)
        
        # self.model.eval()
        self.model.set_mode(False)
        
        with torch.no_grad():
            _, slices = supp_y.sort()
            supp_x, quer_x = supp_x[slices], img

            supp_end = supp_x.size(0)

            x = torch.cat([supp_x, quer_x], dim=0)
            x = self.model(x)
            supp_x, quer_x = x[:supp_end], x[supp_end:]
            supp_x = supp_x.view(5, -1, supp_x.size(-1))
            lg = decode_label(supp_x, quer_x, prob=True)
            return lg

def load_model(config):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'proto_{}'.format(config['global_id']))
    path_to_tmp = os.path.join(path, 'tmp.pt')
    m = torch.load(path_to_tmp)
    device = torch.device(config['device'])
    model = m.to(device)
    return MyMultiManager(model, config)
