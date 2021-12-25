import os
import numpy as np
import random
import torch
import tensorflow as tf
import time
from meta_dataset.data.dataset_spec import BiLevelDatasetSpecification, DatasetSpecification

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tf.random.set_seed(seed)


def to_torch(source, dtype=None) -> torch.Tensor:
    if not isinstance(source, np.ndarray):
        source = source.numpy()
    if dtype is None:
        return torch.from_numpy(source)
    return torch.from_numpy(source).to(dtype)


def to_BCHW(source: torch.Tensor):
    return source.permute(0, 3, 1, 2)


def process_task_batch_numpy(batch, with_origin_label=False):
    # supp  image: batch[0][0]: batch * (WAY * SHOT) * H * W * C
    # supp  label: batch[0][1]: batch * (WAY * SHOT)
    # query image: batch[0][3]: batch * (WAY * QUER) * H * W * C
    # query label: batch[0][4]: batch * (WAY * QUER)

    supp = [batch[0][0].numpy().transpose((0, 1, 4, 2, 3)), batch[0][1].numpy()]
    quer = [batch[0][3].numpy().transpose((0, 1, 4, 2, 3)), batch[0][4].numpy()]
    if with_origin_label:
        supp += [batch[0][2].numpy()]
        quer += [batch[0][5].numpy()]
    
    return supp, quer


def process_task_batch(batch, device=torch.device('cuda:0'), with_origin_label=False, data_augmentor=None):
    # supp  image: batch[0][0]: 1 * (WAY * SHOT) * H * W * C
    # supp  label: batch[0][1]: 1 * (WAY * SHOT)
    # query image: batch[0][3]: 1 * (WAY * QUER) * H * W * C
    # query label: batch[0][4]: 1 * (WAY * QUER)

    supp = [to_torch(batch[0][0])[0].permute(0, 3, 1, 2), to_torch(batch[0][1], dtype=torch.long)[0].to(device)]
    query = [to_torch(batch[0][3])[0].permute(0, 3, 1, 2), to_torch(batch[0][4], dtype=torch.long)[0].to(device)]
    if data_augmentor is not None:
        supp[0] = data_augmentor(supp[0])
        query[0] = data_augmentor(query[0])
    supp[0] = supp[0].to(device)
    query[0] = query[0].to(device)
    if with_origin_label:
        supp += [to_torch(batch[0][2], dtype=torch.long)[0].to(device)]
        query += [to_torch(batch[0][5], dtype=torch.long)[0].to(device)]
    
    return supp, query


def mean(x):
    return sum(x) / len(x)


class pipe_wrap():
    def __init__(self, pipe, process, config, train):
        self.process = process
        self.pipe = pipe
        self.train = train
        self.config = config
    
    def recv(self):
        supp, quer = self.pipe.recv()
        if self.process is not None:
            res = self.process(supp, quer, self.train, self.config)
        else:
            res = [supp, quer]

        return res
    
    def send(self, x):
        self.pipe.send(x)

    def set_path(self, path): pass
    def begin_record(self): pass
    def end_record(self): pass
    def finish(self): pass


class timer():
    def initialize(self, time_begin='auto', time_limit=60 * 100):
        self.time_limit = time_limit
        self.time_begin = time.time() if time_begin == 'auto' else time_begin
        self.time_list = [self.time_begin]
        self.named_time = {}
        return self

    def anchor(self, name=None, end=None):
        self.time_list.append(time.time())
        if name is not None:
            if name in self.named_time:
                if end:
                    assert self.named_time[name]['time_begin'] is not None
                    self.named_time[name]['time_period'].append(self.time_list[-1] - self.named_time[name]['time_begin'])
                else:
                    self.named_time[name]['time_begin'] = self.time_list[-1]
            else:
                assert end == False
                self.named_time[name] = {
                    'time_begin': self.time_list[-1],
                    'time_period': []
                }
        return self.time_list[-1] - self.time_list[-2]

    def query_time_by_name(self, name, method=mean, default=50):
        if name not in self.named_time or self.named_time[name]['time_period'] == []:
            return default
        times = self.named_time[name]['time_period']
        return method(times)

    def time_left(self):
        return self.time_limit - time.time() + self.time_begin
    
    def begin(self, name):
        self.anchor(name, end=False)
    
    def end(self, name):
        self.anchor(name, end=True)
        return self.named_time[name]['time_period'][-1]


def get_base_class_number(data_generator):
    data_meta = data_generator.dataset_spec
    if isinstance(data_meta, BiLevelDatasetSpecification):
        info = data_meta.to_dict()
        train_super_class = info['superclasses_per_split']['TRAIN']
        return sum([info['classes_per_superclass'][x] for x in range(train_super_class)])
    elif isinstance(data_meta, DatasetSpecification):
        info = data_meta.to_dict()
        return info['classes_per_split']['TRAIN']
    raise ValueError('cannot parse class spec file')


def get_root(path = None):
    root = os.path.abspath(os.path.join(os.path.dirname(__name__), '../../'))
    if path is None:
        return root
    return os.path.join(root, path)
