import time
from typing import Dict
t1 = time.time()
import json
import os
from copy import deepcopy
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
os.system('ulimit -HSn 4096')
try:
    import torch
except:
    os.system('conda install --yes pytorch torchvision cudatoolkit=10.1 -c pytorch')
try:
    import lightgbm
except:
    os.system('pip install lightgbm')

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

from src.utils import get_logger

LOGGER = get_logger('GLOBAL')

current_path = os.path.abspath(__file__)
config_file_path = os.path.abspath(os.path.join(os.path.dirname(current_path), 'config.json'))
import json
config = json.load(open(config_file_path, 'r'))

from src.meta.multiprocess import multiprocess
from src.meta.proto import prototype_multi

if config['epochs'] == 'auto':
    config['epochs'] = prototype_multi.MAX_LOCAL_EPOCH

def apply_device(conf: Dict, device, global_id, **kwargs):
    conf['device'] = 'cuda:{}'.format(device)
    conf['global_id'] = global_id
    conf.update(kwargs)
    return conf

multiprocess.GLOBAL_CONFIG = {
    'modules': [prototype_multi, prototype_multi, prototype_multi, prototype_multi],
    'hp': [
        apply_device(deepcopy(config), 0, 0, backbone='resnet50'),
        apply_device(deepcopy(config), 1, 1, backbone='mobilenet'),
        apply_device(deepcopy(config), 2, 2, backbone='wrn50'),
        apply_device(deepcopy(config), 3, 3, backbone='resnet152')
    ],
    'devices': [0,1,2,3],
    'eval_tasks': 200,
    'ensemble': 'all',
    'multiprocess': True or config['multiprocess'],
    'begin_time_stamp': t1
}

LOGGER.info(multiprocess.GLOBAL_CONFIG)

MyMetaLearner = multiprocess.MyMetaLearner
MyLearner = multiprocess.MyLearner
MyPredictor = multiprocess.MyPredictor

t2 = time.time()
LOGGER.info('time used for installing package, set-up gpu', t2 - t1)