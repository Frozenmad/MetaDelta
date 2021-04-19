# specialized for only 4 models
# use file to sync

from copy import deepcopy
import queue
import time
import os
import traceback
import random

import torch
import numpy as np
from src.utils.utils import timer
import torch.multiprocessing as mp
import threading
from torch.multiprocessing import Pipe
from metadl.api.api import MetaLearner, Learner, Predictor
from itertools import cycle
from src.utils.utils import process_task_batch, to_torch
from src.utils import logger
from src.ensemble import GBMEnsembler, GLMEnsembler, NBEnsembler, RFEnsembler
import signal

SIGNALS_TO_NAMES_DICT = dict((getattr(signal, n), n) \
    for n in dir(signal) if n.startswith('SIG') and '_' not in n )

def receive_signal(signum, stack):
    if signum in [1,2,3,15]:
        print('Caught signal %s (%s), exiting.' % (SIGNALS_TO_NAMES_DICT[signum], str(signum)))
    else:
        print('Caught signal %s (%s), ignoring.' % (SIGNALS_TO_NAMES_DICT[signum], str(signum)))

LOGGER = logger.get_logger('main-thread')

GLOBAL_CONFIG = {}

def predict(learner, episode_queue, total_task=200, res='cpu', kargs={}):
    time1 = time.time()
    if 'time_fired' in kargs:
        LOGGER.debug(kargs['taskid'], 'fired time', time1 - kargs['time_fired'])
    # LOGGER.info('task id', kargs['taskid'], 'device', res)
    device = torch.device(res)
    learner.to(device)
    result = []
    for i in range(total_task):
        if isinstance(episode_queue, list):
            supp_x, supp_y, quer_x = episode_queue[i]
        elif hasattr(episode_queue, 'get'):
            supp_x, supp_y, quer_x = episode_queue.get()
        else:
            # LOGGER.info(kargs['taskid'], 'try receiving data', i)
            supp_x, supp_y, quer_x = episode_queue.recv()
            # LOGGER.info(kargs['taskid'], 'receives ', i)
            if i < total_task - 1:
                episode_queue.send(True)
                # LOGGER.info(kargs['taskid'], 'next data query fired')
        res = learner.fit(supp_x, supp_y, quer_x, device=device)
        # LOGGER.info(kargs['taskid'], 'calc on data', i, 'done')
        result.extend(res.tolist())
    learner.to(torch.device('cpu'))
    if 'taskid' in kargs:
        print(kargs['taskid'], time.time() - time1)
    if hasattr(episode_queue,'send'):
        episode_queue.send({'res': result, **kargs})
        return
    return {'res': result, **kargs}

def predict_episode_pipe(learner, data_pipe):
    try:
        while True:
            supp_x, supp_y, img, tim = data_pipe.recv()
            res = learner.fit(supp_x, supp_y, img, device=learner.config['device'])
            data_pipe.send(res)
    except EOFError:
        LOGGER.info('predict episode exit')
    except:
        traceback.print_exc()
    LOGGER.info('anyway, we exit')

def run_exp(module, hp, train, valid, clsnum):
    try:
        mlearner = module(hp)
        mlearner.meta_fit(train, valid, clsnum)
    except:
        LOGGER.info('exp', hp['global_id'], 'terminated with the following error')
        traceback.print_exc()
        LOGGER.info('will exit the experiments')

def ensemble_on_data(args):
    ensembler, reses, labels, name = args
    ensembler._fit(reses, labels)
    t1 = time.time()
    ensembler._fit(reses, labels)
    acc2 = (ensembler._predict(reses).argmax(axis=1) == labels).mean() # meta_ensemble.fc.weight.cpu().tolist()[0]
    t2 = time.time()
    LOGGER.info('ensemble', name, 'acc', acc2, 'time cost', t2 - t1)
    return ensembler, acc2, t2 - t1

class MyMetaLearner(MetaLearner):
    def __init__(self) -> None:
        self.timer = timer().initialize(time_begin=GLOBAL_CONFIG['begin_time_stamp'], time_limit=60 * 110)
        self.__dict__.update(GLOBAL_CONFIG)
        self.total_exp = len(self.modules)
        self.runned_exp_id = 0
        LOGGER.info('initialization done')
        
    def meta_fit(self, meta_dataset_generator):


        catchable_sigs = set(signal.Signals) - {signal.SIGKILL, signal.SIGSTOP}
        for sig in catchable_sigs:
            signal.signal(sig, receive_signal)  # Substitute handler of choice for `print`
        LOGGER.debug('My PID: %s' % os.getpid())

        self.timer.begin('main training')
        mp.set_start_method('spawn', force=True)
        
        # >>> BUG: OS Error: Too many opened files
        # >>> SOLVED: by `ulimit -HSn 4096`
        # Now, we change all the queues to pipe
        self.timer.begin('build data pipeline')
        # every 10 epoch will produce one valid
        train_data_reservoir = [queue.Queue(32 * 10) for i in range(len(self.devices))]
        valid_data_reservoir = [queue.Queue(200) for i in range(len(self.devices))]
        meta_valid_reservoir = [queue.Queue(self.eval_tasks) for i in range(self.total_exp)]
        train_recv, valid_recv = [], []
        train_send, valid_send = [], []
        for i in range(len(self.devices)):
            recv, send = Pipe(True)
            # activate the first handshake
            recv.send(True)
            train_recv.append(recv)
            train_send.append(send)
            recv, send = Pipe(True)
            # activate the first handshake
            recv.send(True)
            valid_recv.append(recv)
            valid_send.append(send)

        def apply_device_to_hp(hp, device):
            if isinstance(device, str):
                hp['device'] = device
            else:
                hp['device'] = 'cuda:{}'.format(device)
            return hp
        
        self.timer.end('build data pipeline')

        self.timer.begin('build main proc pipeline')
        clsnum = self.class_number
        LOGGER.info('base class number detected', clsnum)
        procs = [mp.Process(
            target=run_exp,
            args=(self.modules[i].MyMetaLearner, apply_device_to_hp(self.hp[i], dev), train_recv[i], valid_recv[i], clsnum) 
        ) for i, dev in enumerate(self.devices)]
        for p in procs: p.daemon = True; p.start()

        self.timer.end('build main proc pipeline')
        LOGGER.info('build data', self.timer.query_time_by_name('build data pipeline'), 'build proc', self.timer.query_time_by_name('build main proc pipeline'))
        label_meta_valid = []

        data_generation = True

        self.timer.begin('prepare dataset')
        meta_train_dataset = meta_dataset_generator.meta_train_pipeline.batch(1)
        meta_train_generator = cycle(iter(meta_train_dataset))
        meta_valid_dataset = meta_dataset_generator.meta_valid_pipeline.batch(1)
        meta_valid_generator = cycle(iter(meta_valid_dataset))
        self.timer.end('prepare dataset')
        LOGGER.info('prepare dataset', self.timer.query_time_by_name('prepare dataset'))

        valid_ens_data_load_number = 0

        def generate_data():
            # manage data globally
            while data_generation:
                for i in range(32 * 10):
                    # load train
                    if not data_generation:
                        break
                    data_train = process_task_batch(next(meta_train_generator), device=torch.device('cpu'), with_origin_label=True)
                    for dr in train_data_reservoir:
                        try:
                            dr.put_nowait(data_train)
                        except:
                            pass
                    time.sleep(0.0001)

                for i in range(200):
                    # load valid
                    if not data_generation:
                        break
                    data_valid = process_task_batch(next(meta_valid_generator), device=torch.device('cpu'), with_origin_label=False)
                    for dr in valid_data_reservoir:
                        try:
                            dr.put_nowait(data_valid)
                        except:
                            pass
                    if random.random() < 0.1:
                        for dr in meta_valid_reservoir:
                            try:
                                if dr.qsize() < self.eval_tasks:
                                    valid_ens_data_load_number += 1
                                    dr.put_nowait([data_valid[0][0], data_valid[0][1], data_valid[1][0]])
                                    label_meta_valid.extend(data_valid[1][1].tolist())
                            except:
                                pass
                    time.sleep(0.0001)

        def put_data_train_passive(i):
            while data_generation:
                try:
                    if train_send[i].recv():
                        supp, quer = train_data_reservoir[i].get()
                        data = self.modules[i].process_data(supp, quer, self.hp[i], True)
                        train_send[i].send(data)
                    else:
                        return
                except:
                    pass

        def put_data_valid_passive(i):
            while data_generation:
                try:
                    if valid_send[i].recv():
                        supp, quer = valid_data_reservoir[i].get()
                        data = self.modules[i].process_data(supp, quer, self.hp[i], False)
                        valid_send[i].send(data)
                    else:
                        return
                except:
                    pass
        
        thread_pool = [threading.Thread(target=generate_data)] + \
            [threading.Thread(target=put_data_train_passive, args=(i,)) for i in range(self.total_exp)] + \
            [threading.Thread(target=put_data_valid_passive, args=(i,)) for i in range(self.total_exp)]
        
        for th in thread_pool: th.daemon = True; th.start()

        try:
            # we leave about 20 min for decoding of test
            for p in procs: p.join(max(self.timer.time_left() - 60 * 10, 0.1))
        
            self.timer.begin('clear env')
            # terminate proc that is out-of-time
            LOGGER.info('Main meta-train is done', '' if self.timer.time_left() > 60 else 'time out exit')
            LOGGER.info('time left', self.timer.time_left(), 's')
            for p in procs:
                if p.is_alive():
                    p.terminate()
            
            data_generation = False
            # in case there are blocking
            for q in train_data_reservoir + valid_data_reservoir:
                if q.empty():
                    q.put(False)
            for s in train_recv + valid_recv: s.send(False)
            for s in train_send + train_recv + valid_send + valid_recv: s.close()
            for p in thread_pool: p.join()
            self.timer.end('clear env')
            LOGGER.info('clear env', self.timer.query_time_by_name('clear env'))
            
            self.timer.end('main training')
        except Exception:
            LOGGER.info('error occured in main process')
            traceback.print_exc()

        LOGGER.info('spawn total {} meta valid tasks. main training time {}'.format(valid_ens_data_load_number, self.timer.query_time_by_name('main training')))
        
        self.timer.begin('load learner')

        self.meta_learners = [None] * self.total_exp

        def load_model(args):
            module, hp, i = args
            self.meta_learners[i] = module.load_model(hp)

        pool = [threading.Thread(target=load_model, args=((self.modules[i], self.hp[i], i), )) for i in range(self.total_exp)]
        for p in pool: p.daemon=True; p.start()
        for p in pool: p.join()

        self.timer.end('load learner')
        LOGGER.info('load learner done, time spent', self.timer.query_time_by_name('load learner'))
        
        if not isinstance(self.ensemble, int):
            # instead of just weighted sum, we plan to use stacking
            procs = []
            reses = [None] * len(self.meta_learners)
            
            self.timer.begin('validation')
            
            recv_list, sent_list = [], []
            for i in range(self.total_exp):
                r, s = Pipe(True)
                r.send(True)
                recv_list.append(r)
                sent_list.append(s)

            pool = mp.Pool(self.total_exp)
            procs = pool.starmap_async(predict, [
                (self.meta_learners[i], 
                recv_list[i], 
                self.eval_tasks, 
                self.hp[i]['device'],
                {
                    'time_fired': time.time(),
                    'taskid': i
                }
            ) for i in range(self.total_exp)])
            
            # start sub thread to pass data
            def pass_meta_data(i):
                for _ in range(self.eval_tasks):
                    if sent_list[i].recv():
                        # LOGGER.info(i, 'fire data signal get')
                        sent_list[i].send(meta_valid_reservoir[i].get())
                        # LOGGER.info(i, 'data is sent')
            
            threads = [threading.Thread(target=pass_meta_data, args=(i, )) for i in range(self.total_exp)]
            for t in threads: t.daemon = True; t.start()

            for _ in range(self.eval_tasks - valid_ens_data_load_number):
                data_valid = next(meta_valid_generator)
                data_valid = process_task_batch(data_valid, device=torch.device('cpu'), with_origin_label=False)
                label_meta_valid.extend(data_valid[1][1].tolist())
                for dr in meta_valid_reservoir:
                    dr.put([data_valid[0][0], data_valid[0][1], data_valid[1][0]])
                # LOGGER.info('put data!')
            # LOGGER.info('all data done!')
            
            # now we can receive data
            for t in threads: t.join()
            reses = [sent_list[i].recv()['res'] for i in range(self.total_exp)]
            # every res in reses is a np.array of shape (eval_task * WAY * QUERY) * WAY
            ENS_VALID_TASK = 50
            ENS_VALID_ELEMENT = ENS_VALID_TASK * 5 * 19
            reses_test_list = [deepcopy(res[-ENS_VALID_ELEMENT:]) for res in reses]

            self.timer.end('validation')
            LOGGER.info('valid data predict done', self.timer.query_time_by_name('validation'))
            
            weight = [1.] * len(self.meta_learners)
            labels = np.array(label_meta_valid, dtype=np.int)                            # 19000
            acc_o = ((np.array(weight)[:,None, None] / sum(weight) * np.array(reses)).sum(axis=0).argmax(axis=1) == labels).mean()
            reses = np.array(reses, dtype=np.float).transpose((1, 0, 2))
            reses_test = reses[-ENS_VALID_ELEMENT:].reshape(ENS_VALID_ELEMENT, -1)
            reses = reses[:-ENS_VALID_ELEMENT]
            reses = reses.reshape(len(reses), -1)
            labels_test = labels[-ENS_VALID_ELEMENT:]
            labels = labels[:-ENS_VALID_ELEMENT]
            LOGGER.info('voting result', acc_o)

            self.timer.begin('ensemble')

            # mp.set_start_method('fork', True)
            result = pool.map(ensemble_on_data, [
                (GBMEnsembler(), reses, labels, 'gbm'),
                (GLMEnsembler(), reses, labels, 'glm'),
                (NBEnsembler(), reses, labels, 'nb'),
                (RFEnsembler(), reses, labels, 'rf') # too over-fit on simple dataset
            ])

            # test the ensemble model
            def acc(logit, label):
                return (logit.argmax(axis=1) == label).mean()
            res_test = [x[0]._predict(reses_test) for x in result]
            acc_test = [acc(r, labels_test) for r in res_test]
            acc_single_test = [acc(np.array(r), labels_test) for r in reses_test_list]
            LOGGER.info('ensemble test', 'gbm', 'glm', 'nb', 'rf', acc_test)
            LOGGER.info('single test', acc_single_test)

            if max(acc_test) > max(acc_single_test):
                LOGGER.info("will use ensemble model")
                #idx_acc_max = np.argmax([x[1] for x in result])
                idx_acc_max = np.argmax(acc_test)
                self.timer.end('ensemble')
                print('best ensembler', ['gbm', 'glm', 'nb', 'rf'][idx_acc_max], 'acc', acc_test[idx_acc_max])
                print('ensemble done, time cost', self.timer.query_time_by_name('ensemble'))

                # currently we use mean of output as ensemble
                return MyLearner(self.meta_learners, result[idx_acc_max][0], timers=self.timer)
            else:
                LOGGER.info("will use single model")
                idx_acc_max = np.argmax(acc_single_test)
                self.timer.end('ensemble')
                print('best single model id', idx_acc_max)
                print('ensemble done, time cost', self.timer.query_time_by_name('ensemble'))

                # return only the best meta learners
                return MyLearner([self.meta_learners[idx_acc_max]], 0, self.timer)
        return MyLearner([self.meta_learners[self.ensemble]], 0, timers=self.timer)

# change the logic to put all the supp img, lab to MyPredictor
# there is a method that we can start the process as long as the load() is exececuted
class MyLearner(Learner):
    def __init__(self, meta_learners=None, meta_ensemble=None, timers=None) -> None:
        self.__dict__.update(GLOBAL_CONFIG)
        self.timer = timers
        self.learner = meta_learners
        self.meta_ensemble = meta_ensemble
        self.loaded = False
        self.epoch = 0
    
    def fit(self, dataset_train):
        img, lab = None, None
        if (self.epoch % 50 == 0) and self.epoch > 0:
            LOGGER.info('mean testing speed', self.timer.query_time_by_name('predict'), 'estimated max time left for one epoch', self.timer.time_left() / max((600 - self.epoch), 1))
        self.epoch += 1
        for idx, (image, label) in enumerate(dataset_train):
            img = to_torch(image).permute(0, 3, 1, 2)
            lab = to_torch(label, dtype=torch.long)

        # return MyMultiPredictor(self.learner, self.meta_ensemble, (img, lab))
        return MyMultiPredictor([self.proc, self.send_list, self.recv_list], self.meta_ensemble, (img, lab), self.timer)

    def save(self, path_to_save):
        t1 = time.time()
        def save_sub_model(i):
            os.makedirs(os.path.join(path_to_save, str(i)), exist_ok=True)
            self.learner[i].save(os.path.join(path_to_save, str(i)))
        
        threads = [threading.Thread(target=save_sub_model, args=(i,)) for i in range(len(self.learner))]
        for t in threads: t.daemon = True; t.start()

        torch.save({
            'weight': self.meta_ensemble,
            'num_learner': len(self.learner),
            'time_begin': self.timer.time_begin
        }, os.path.join(path_to_save, 'hp.pt'))

        for t in threads: t.join()
        t2 = time.time()
        LOGGER.info('save time', t2 - t1)
            
    def load(self, path_to_model):
        if not self.loaded:
            t1 = time.time()
            data = torch.load(os.path.join(path_to_model, 'hp.pt'))
            self.meta_ensemble = data['weight']
            self.learner = [None] * data['num_learner']
            self.timer = timer().initialize(time_begin=data['time_begin'], time_limit=60 * 110)
            LOGGER.info('time left for test', self.timer.time_left())
            def load_sub_model(i):
                module = self.modules[i]
                learner = module.MyMultiManager(self.hp[i], None)
                learner.load(os.path.join(path_to_model, str(i)))
                self.learner[i] = learner
            pool = [threading.Thread(target=load_sub_model, args=(i,)) for i in range(data['num_learner'])]
            for t in pool: t.daemon = True; t.start()
            for t in pool: t.join()
            t2 = time.time()
            LOGGER.info('load time', t2 - t1)
            mp.set_start_method('spawn', True)
            self.send_list = []
            self.recv_list = []
            for i in range(len(self.learner)):
                r, s = Pipe(True)
                self.send_list.append(s)
                self.recv_list.append(r)
            self.proc =  []
            for i in range(len(self.learner)):
                oric = mp.Process(target=predict_episode_pipe, args=(self.learner[i], self.recv_list[i]))
                oric.daemon = True
                oric.start()
                self.proc.append(oric)
            self.loaded = True


class MyPredictor(Predictor):
    def __init__(self, predictor: list, meta_ensemble, supp, timers) -> None:
        self.predictor = predictor
        self.ens = meta_ensemble
        self.supp = supp
        self.timer = timers
    
    def predict(self, dataset_test):
        for image in dataset_test:
            result = []
            image = image[0]
            # 95 * 28 * 28 * 3
            image = to_torch(image).permute(0, 3, 1, 2)

            # make predictions
            for p in self.predictor:
                result.append(p.fit(self.supp[0], self.supp[1], image))
            result = np.array(result).transpose((1, 0, 2)).reshape(len(result[0]), -1)
            result = self.ens.predict_log_proba(result)
            return result

class MyMultiPredictor:
    def __init__(self, predictors, meta_ensemble, supp, timers) -> None:
        self.__dict__.update(GLOBAL_CONFIG)
        procs, data_pipe, out_pipe = predictors
        self.procs = procs
        self.data_pipe = data_pipe
        self.out_pip = out_pipe
        self.ens = meta_ensemble
        self.supp = supp
        self.timer = timers
    
    def predict(self, dataset_test):
        self.timer.begin('predict')
        for image in dataset_test:
            image = image[0]
            # 95 * 28 * 28 * 3
            image = to_torch(image).permute(0, 3, 1, 2)

            data = [self.supp[0], self.supp[1], image]
            for i in range(len(self.procs)):
                self.data_pipe[i].send(data + [time.time()])
            
            result = [self.data_pipe[i].recv() for i in range(len(self.procs))]
            if isinstance(self.ens, int):
                self.timer.end('predict')
                return result[self.ens]

            result = np.array(result).transpose((1, 0, 2)).reshape(len(result[0]), -1)
            result = self.ens._predict(result)
            self.timer.end('predict')
            return result
