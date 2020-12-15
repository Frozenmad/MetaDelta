import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .base_multi import ProtoMetaLearner, ProtoMultiManager
from ...learner.pretrained_encoders.resnet_pretrained import resnet152, resnet50, wide_resnet50_2
from ...learner.pretrained_encoders.mobilenet_pretrained import mobilenet_v2
from src.utils import logger
import time

def mean(x):
    return sum(x) / len(x)

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def distance_label_propagation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
    return logits

def make_protomap(support_set, way):
    B, D = support_set.shape
    shot = B // way
    protomap = support_set.reshape(shot, way, D)
    protomap = protomap.mean(dim=0)

    return protomap


def flatten(sets):
    # flatten
    sets = torch.flatten(sets, start_dim=1)
    sets = F.normalize(sets)

    return sets

def add_query(support_set, query_set, prob, way):

    B, D = support_set.shape
    shot = B // way
    per_class = support_set.reshape(shot, way, D)

    protomap = []
    for i in range(way):
        ith_prob = prob[:, i].reshape(prob.size(0), 1)
        ith_map = torch.cat((per_class[:, i], query_set * ith_prob), dim=0)
        ith_map = torch.sum(ith_map, dim=0, keepdim=True) / (ith_prob.sum() + shot)
        protomap.append(ith_map)

    return torch.cat(protomap, dim=0)

def param_free_distance_cal(query, supp):
    query, supp = normalize(query), normalize(supp)
    logit = distance_label_propagation(query, supp)
    return F.softmax(logit, dim=-1)

def mct_label_propagation(query, supp, cal_distance=param_free_distance_cal, shot=1, iters=11):
    prob_list = []
    way = len(supp) // shot
    for iter in range(iters):
        # Make Protomap
        if iter == 0:
            protomap = make_protomap(supp, way)
        else:
            protomap = add_query(supp, query, prob_list[iter-1], way)

        prob = cal_distance(query, protomap)
        prob_list.append(prob)
    return torch.log(prob_list[-1] + 1e-6)

gconfig = {
    'device': 'cuda:0',
    'pretrain_lr': 0.0001,
    'eval_epoch': 10,
    'eval_tasks': 200,
    'batch_size': 32,
    'lr': 0.0001,
    'epochs': 20000,
    'patience': 20,
    'clip_norm': 1.0,
    'use_pretrain': True,
    'pretrain_epoch': 20000,
    'cls_type': 'linear',
    'train_way': 5,
    'pretrain_shot': 2,
    'first_eval': 0,
    'rotation': True,   # pretrain with 'rotation' loss (in Phase #2)
    'way': 32,    # TODO: keep consistent with config.gin
    "global_id": 0
}

def normalize(emb):
    emb = emb / emb.norm(dim=1)[:,None]
    return emb

def process_data(supp, query, train=True, config=gconfig):
    if train:
        # return [supp, query]
        # load train data
        way, number = len(supp[0]), len(query[0]) // len(supp[0]) + 1
        others = supp[0].size()[1:]
        if config['pretrain_shot'] == 1:
            x = supp[0]
            y = supp[2]
        else:
            x = torch.cat([supp[0], query[0]])
            y = torch.cat([supp[2], query[2]])
            y, slices = y.sort()
            x = x[slices].reshape(way, number, *others)
            y = y.reshape(way, number)
            randidx = torch.randperm(number)[:config['pretrain_shot']]
            x, y = x[:,randidx,:].reshape(way * config['pretrain_shot'], *others), y[:,randidx].reshape(-1)
        
        if config['rotation']:
            # x.shape in CIFAR100: [64=32way*2shot, 3, 28, 28]
            x90 = torch.rot90(x, 1, [2, 3])
            x180 = torch.rot90(x90, 1, [2, 3])
            x270 = torch.rot90(x180, 1, [2, 3])
            x_ = torch.cat((x, x90, x180, x270), 0)
            y_ = torch.cat((y, y, y, y), 0)
            x = x_
            y = y_
        return [x, y]
    else:
        # load valid data
        return [supp, query]

MODEL = {
    'resnet50': resnet50,
    'mobilenet': mobilenet_v2,
    'wrn50': wide_resnet50_2,
    'resnet152': resnet152
}

def decode_label(sx, qx, label_prop, prob=True):
    sx = normalize(sx)
    qx = normalize(qx)

    if label_prop == 'mct':
        lg = mct_label_propagation(qx, sx)
    else:
        lg = distance_label_propagation(qx, sx)
    if prob:
        lg = F.softmax(lg, dim=1)
    lg = lg.detach().cpu().numpy()

    return lg

class MyMetaLearner(ProtoMetaLearner):
    def __init__(self, config=gconfig) -> None:
        self.__dict__.update(config)
        self.config = config
        self.logger = logger.get_logger('proto_{}'.format(self.global_id))

        super().__init__(self.epochs, self.eval_epoch, self.patience, self.eval_tasks, self.batch_size, self.first_eval, self.logger)
        self.device = torch.device(self.device)
        self.logger.info('current hp', config)

    def load_model(self):
        load_model(self.config)

    def create_model(self, class_num):
        self.timer.begin('load pretrained model')
        self.model = MODEL[self.backbone](True, True)
        self.model.to(self.device)
        # for origin class training
        times = self.timer.end('load pretrained model')
        self.logger.info('current model', self.model)
        self.logger.info('load time', times, 's')
        self.dim = self.model(torch.randn(2,3,28,28).to(self.device)).size()[-1]
        self.logger.info('detect encoder dimension', self.dim)
        if self.use_pretrain:
            # manually extract the dim
            self.cls = nn.Linear(self.dim, class_num).to(self.device)
            if self.rotation:
                self.rotate_cls = nn.Sequential(nn.Linear(self.dim, 4)).to(self.device)
                self.rotate_label = torch.tensor(np.arange(4)).unsqueeze(dim=1).repeat(1, self.way * self.pretrain_shot).flatten().to(self.device)
                self.opt_pretrain = optim.Adam(list(self.model.parameters()) + list(self.cls.parameters()) + 
                                               list(self.rotate_cls.parameters()), lr=self.pretrain_lr)
            else:
                self.opt_pretrain = optim.Adam(list(self.model.parameters()) + list(self.cls.parameters()), lr=self.pretrain_lr)

        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)
        self.best_model = None
    
    def on_train_begin(self, epoch):
        self.model.train()
        self.cls.train()
        # if self.pretrain_strategy == 'rotation':
        if self.rotation:
            self.rotate_cls.train()
        self.err_list = []
        self.acc_list = []
        self.opt.zero_grad()
        self.opt_pretrain.zero_grad()
        self.mode = 'pretrain'
        return True

    def on_train_end(self, epoch):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad /= len(self.err_list)
        if self.rotation:
            nn.utils.clip_grad.clip_grad_norm_(list(self.model.parameters()) + list(self.cls.parameters()) + 
                                                list(self.rotate_cls.parameters()), max_norm=self.clip_norm)
        self.opt_pretrain.step()
        err = mean(self.err_list)
        acc = mean(self.acc_list)
        self.logger.info('epoch %2d mode: %s error %.6f acc %.6f' % (epoch, self.mode, err, acc))
    
    def mini_epoch(self, train_pipe, epoch, iters):
        # use pretrain
        x, y = train_pipe.recv()
        train_pipe.send(True)
        x = x.to(self.device)
        y = y.to(self.device)
        feature = self.model(x)
        logit = self.cls(feature)
        loss = F.cross_entropy(logit, y)
        if self.rotation:
            rotate_logit = self.rotate_cls(feature)
            rloss = F.cross_entropy(rotate_logit, self.rotate_label)
            loss = 0.5 * loss + 0.5 * rloss
        loss.backward()
        self.err_list.append(loss.item())
        self.acc_list.append(accuracy(logit, y))
    
    def eval_one_episode(self, valid_pipe):
        with torch.no_grad():
            t1 = time.time()
            supp, query = valid_pipe.recv()
            valid_pipe.send(True)
            t2 = time.time()
            _, slices = supp[1].to(self.device).sort()
            supp_x = self.model( supp[0].to(self.device)[slices])
            quer_x = self.model(query[0].to(self.device))
            #quer_x, supp_x = embeding_propagation(quer_x, supp_x)
            t3 = time.time()
            logit = decode_label(supp_x, quer_x, 'euclidean')
            t4 = time.time()
            if isinstance(logit, torch.Tensor):
                logit = np.array(logit.detach().cpu())
            acc = (logit.argmax(1) == np.array(query[1])).mean()
            t5 = time.time()
            def add_ele(name, ele):
                if name in self.time_dict:
                    self.time_dict[name].append(ele)
                else:
                    self.time_dict[name] = [ele]
            add_ele('com_time', t2 - t1)
            add_ele('model_time', t3 - t2)
            add_ele('transform_time', t4 - t3)
            add_ele('acc_time', t5 - t4)
        return acc

    def on_eval_begin(self, epoch):
        self.model.eval()
        self.time_dict = {}
        return True
    
    def on_eval_end(self, epoch, acc, patience_now):
        self.logger.info('= time statistic =' , \
            sum(self.time_dict['com_time']), sum(self.time_dict['model_time']), \
            sum(self.time_dict['transform_time']), sum(self.time_dict['acc_time']))
        return True

    def save_model(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'proto_{}'.format(self.global_id))
        os.makedirs(path, exist_ok=True)
        torch.save({
            'model': self.model
        }, os.path.join(path, 'tmp.pt'))
    
    def make_learner(self):
        return MyMultiManager(self.model)

class MyMultiManager(ProtoMultiManager):
    def __init__(self, model=None, config=gconfig) -> None:
        self.model = model
        self.config = config
        self.loaded = False
        
    def load_model(self, path):
        if self.loaded:
            return
        self.model = torch.load(os.path.join(path, 'model.pt'))
        other = torch.load(os.path.join(path, 'config.pt'))
        self.config = other['config']
        self.loaded = True

    def save_model(self, path):
        torch.save(self.model, os.path.join(path, 'model.pt'))
        torch.save({
            'config': self.config,
        }, os.path.join(path, 'config.pt'))
    
    def to(self, device):
        self.model.to(device)

    def eval_one_episode(self, supp_x, supp_y, img, device):
        self.model.to(device)
        supp_x = supp_x.to(device)
        supp_y = supp_y.to(device)
        img = img.to(device)
        self.model.eval()

        with torch.no_grad():
            _, slices = supp_y.sort()
            supp_x = self.model(supp_x[slices])
            quer_x = self.model(img)
            lg = decode_label(supp_x, quer_x, 'mct', prob=True)
            return lg

def load_model(config=gconfig):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'proto_{}'.format(config['global_id']))
    path_to_tmp = os.path.join(path, 'tmp.pt')
    m = torch.load(path_to_tmp)
    device = torch.device(gconfig['device'])
    model = m['model'].to(device)
    return MyMultiManager(model, config)