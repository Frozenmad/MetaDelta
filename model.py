import time
from typing import Dict
t1 = time.time()
import json
import os
from copy import deepcopy

# os.system('pip3 install -U torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html')
# os.system('pip install numpy cython==0.29.24')
# os.system('pip install POT==0.7.0')
# os.system('pip install dill==0.3.4')
# os.system('pip install tqdm==4.62.2 lightgbm==3.2.1')
# os.system('pip install timm')

import tensorflow as tf

from src.utils import get_logger
from src.utils.utils import set_seed

set_seed(1234)

LOGGER = get_logger('GLOBAL')

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

current_path = os.path.abspath(__file__)
config_file_path = os.path.abspath(os.path.join(os.path.dirname(current_path), 'config.json'))
config_gin_path = os.path.abspath(os.path.join(os.path.dirname(current_path), 'config.gin'))
import json
config = json.load(open(config_file_path, 'r'))

from src.meta.multiprocess import multiprocess as controller
from src.meta.proto import prototype_multi

def apply_device(conf: Dict, device, global_id, **kwargs):
    conf['device'] = 'cuda:{}'.format(device)
    conf['global_id'] = global_id
    conf.update(kwargs)
    return conf

controller.GLOBAL_CONFIG = {
    'modules': [prototype_multi], # , prototype_multi],# prototype_multi],
    'hp': [
        # apply_device(deepcopy(config), 0, 0, backbone='enet', name='tf_efficientnetv2_s_in21k', size=300, first_eval=-1, epochs=1000, parameters=[6,7], momentum=0),
        # apply_device(deepcopy(config), 2, 2, backbone='enet', name='tf_efficientnet_b0_ns', size=224, first_eval=-1, epochs=1000, parameters=[5,6,7,8], momentum=0.01),
        apply_device(deepcopy(config), 0, 1, backbone='rn', name='swsl_resnet50', size=224, first_eval=-1, epochs=1000, parameters=[3,4], momentum=0.1),
    ],
    'devices': [0], #,0],
    'eval_tasks': 250,
    'ensemble': "all",
    'begin_time_stamp': t1,
    "train_cache_size": 40,
    "valid_cache_size": 40,
    "process_protocol": "process-in-main",
    "fix_valid": True
}

LOGGER.info(controller.GLOBAL_CONFIG)

MyMetaLearner = controller.MyMetaLearner
MyLearner = controller.MyLearner
MyPredictor = controller.MyMultiPredictor

t2 = time.time()
LOGGER.info('time used for installing package, set-up gpu', t2 - t1)
