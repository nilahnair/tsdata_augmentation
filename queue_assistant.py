'''
Created on Jun 13, 2019

@author: fwolf
'''

import argparse
import io
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import pymongo
import yaml
from pymongo import MongoClient
from sacred.observers import (FileStorageObserver, MongoObserver,
                              QueuedMongoObserver)


from sacred import SETTINGS

def configure_sacred():
    SETTINGS.DISCOVER_SOURCES = 'dir'
    SETTINGS.DISCOVER_DEPENDENCIES = 'sys'

def load_credentials(path='~/.mongodb_readonly_credentials'):
    path = os.path.expanduser(path)

    logger = logging.getLogger('::load_credentials')
    logger.info(f'Loading credientials from {path}')
    with io.open(path) as f:
        user, pw, url, db_name = f.read().strip().split(',')

    return user, pw, url, db_name

class QAssistant(object):
    '''
    classdocs
    '''

    def __init__(self,
        port='27017',
        visible_gpus=[0],
        visible_exps=None,
        max_used_mem=3,
        ignore_ready_check=False,
        readonly=False,
        filter_by_hostname=True,
        logdir='queue_assistant_logs/'):
        '''
        Constructor
        '''
        logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)


        self.visible_gpus = visible_gpus
        self.visible_exps = visible_exps
        self.max_used_mem = max_used_mem
        self.ignore_ready_check = ignore_ready_check
        self.filter_by_hostname = filter_by_hostname

        
        self.port = port

        self.user, self.pw, self.url, self.db_name = load_credentials(path='~/.mongodb_readonly_credentials' if readonly else '~/.mongodb_credentials')
        mongo_client = MongoClient(host=f'{self.url}:{self.port}',
                                   username=self.user,
                                   password=self.pw,
                                   authSource='admin',
                                   authMechanism='SCRAM-SHA-1'
                                   )

        self.db = mongo_client[self.db_name]
        logdir = Path(logdir)
        logdir.mkdir(exist_ok=True)
        self.logdir = str(logdir)
        
    def _get_free_gpus(self, max_used_percent):
        logger = logging.getLogger('QAssistant::get_free_gpus')

        # gpu_info = nvgpu.gpu_info()
        # free_gpus = [gpu for gpu in self.visible_gpus if gpu_info[gpu]['mem_used_percent'] < 3]
        # return free_gpus
    
        from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, \
            nvmlDeviceGetMigDeviceHandleByIndex, nvmlDeviceGetUUID, nvmlDeviceGetMemoryInfo
        
        nvmlInit()

        free_gpu_uuids = []

        for id in self.visible_gpus:
            if '_' in str(id): # parse as mig mode!
                gpu_id, mig_id = (int(i) for i in id.split('_'))
                logger.info(f'Parsing {id} in MIG-Mode (GPU {gpu_id}, MIG Dev {mig_id})')
                device = nvmlDeviceGetHandleByIndex(index=gpu_id)
                device = nvmlDeviceGetMigDeviceHandleByIndex(device, index=mig_id)
            else:
                device = nvmlDeviceGetHandleByIndex(index=int(id))

            uuid = nvmlDeviceGetUUID(device)
            mem_info = nvmlDeviceGetMemoryInfo(device)
            mem_used = mem_info.used * 9.5367e-7 # in MiB
            mem_total = mem_info.total * 9.5367e-7 # in MiB
            used_percent = (mem_info.used / mem_info.total) * 100
            logger.info(f'Device {uuid} uses {mem_used / 1024:.1f} GiB / {mem_total / 1_024:.1f} GiB of memory ({used_percent:.2f}% used)')
        
            if used_percent < max_used_percent: # gpu is ready to use!
                logger.info(f'Adding {uuid} to list of free gpus')
                free_gpu_uuids.append({'uuid': uuid, 'mem_used': mem_used, 'mem_total': mem_total})

        nvmlShutdown()

        return free_gpu_uuids
         
    def _load_experiment(self, _id):
        return self.db.runs.find_one({'_id': _id})
    
    def _get_queued_experiment_ids(self): 
        queue = self.db["runs"].find({'status': 'QUEUED'}).sort('config.pretrained_id', pymongo.DESCENDING)
        return [exp['_id'] for exp in queue]
    
    def _filter_ready_experiment_ids(self, queued_ids):
        logger = logging.getLogger('QAssistant::filter_ready_experiments')

        filtered_ids = queued_ids.copy()

        for exp_id in queued_ids:
            logger.info(f'Checking experiment {exp_id}')
            exp = self._load_experiment(exp_id)

            # exp_path = exp['config']['fs_observer_path'] #TODO: check if experiment path exists, avoid constantly failing experiments

            if self.filter_by_hostname:
                local_hostname = os.uname()[1]
                exp_hostname = exp['host']['hostname']
                if local_hostname != exp_hostname:
                    filtered_ids.remove(exp_id)
                    logger.info(f'Skipping {exp_id} - Experiment was queued on a different host (local hostname {local_hostname} != experiment hostname {exp_hostname})')
                    continue

            # check if exp depends on a pretrained_id
            if 'pretrained_id' in exp['config']:
                pretrained_id = exp['config']['pretrained_id']

                logger.info(f'Experiment depends on {pretrained_id}')

                #check if pretrained_id exp is completed
                pretrained_path = Path(exp['config']['fs_observer_path']) / str(pretrained_id)

                if not pretrained_path.is_dir():
                    logger.info(f'Skipping {exp_id} - folder {pretrained_path} does not exist')
                    filtered_ids.remove(exp_id)

                elif not (pretrained_path / 'last.ckpt').is_file():
                    logger.info(f'Skipping {exp_id} - checkpoint for {pretrained_id} does not exist')
                    filtered_ids.remove(exp_id)

                elif not (pretrained_path / 'config.json').is_file():
                    logger.info(f'Skipping {exp_id} - config.json for {pretrained_id} does not exist')
                    filtered_ids.remove(exp_id)

                else:
                    # check if status is COMPLETED
                    with open(pretrained_path / 'run.json', 'r') as p:
                        parent_run = json.load(p)
                        if not parent_run['status'] in ['COMPLETED', 'FAILED']: # hack, sometimes pretraining fails in test phase, but model itself is actually trained
                            logger.info(f'Skipping {exp_id} - {pretrained_id} is not completed yet')
                            filtered_ids.remove(exp_id)
    
        return filtered_ids

    def _delete_sources(self, _id, temp_dir=Path.home() / '.sacred_tmp'):
        logger = logging.getLogger('QAssistant::clean_sources')
        root_dir = temp_dir / str(_id)
        if root_dir.exists():
            logger.info(f'Removing old sources from {root_dir}')
            shutil.rmtree(str(root_dir))

    def _load_sources(self, _id, temp_dir=Path.home() / '.sacred_tmp', clean=True):
        logger = logging.getLogger('QAssistant::load_sources')

        if clean:
            self._delete_sources(_id, temp_dir)

        root_dir = (temp_dir / str(_id))
        root_dir.mkdir(parents=True, exist_ok=True)

        run = self._load_experiment(_id)
        
        # download sources
        for filename, file_id in run['experiment']['sources']:
            logger.info(f'Downloading {filename}')
            # create dir
            filedir = (root_dir / filename).parent
            filedir.mkdir(parents=True, exist_ok=True)

            if '__init__' in filename: # create __init__.py by hand, these are None in mongodb
                (root_dir / filename).touch()
                continue

            file_str = self.db.fs.chunks.find_one({'files_id': file_id})['data']

            with open(root_dir/filename, 'wb') as f:
                f.write(file_str)
        
        # download config
        logger.info('Downloading config')
        config = run['config']
        with open(root_dir / 'config.yaml', 'w') as c:
            yaml.dump(config, c)
        
        return str(root_dir)

    def _get_sacred_exp(self, mainfile):
        if mainfile == 'main.py':
            from main import ex
        else:
            raise ValueError(f'Unknown mainfile {mainfile}')
        
        return ex
    
    # def _start_experiment(self, queued_exp, gpu_id, replace=True):
    def _start_experiment(self, queued_exp, replace=True, queue=False):
        ex = self._get_sacred_exp(queued_exp['experiment']['mainfile']) 

        if not queue:
            old_status = queued_exp['status']
            queued_exp['status'] = 'INITIALIZING'

        # queued_exp['config']['gpus'] = [0] #TODO: able to pass mutliple IDs

        if replace == True:
            self.db.runs.replace_one({'_id': queued_exp['_id'], 'status': old_status},
                                        replacement=queued_exp)
        
            ex.observers[0] = MongoObserver(url=self.url,
                                            db_name=self.db_name,
                                            username=self.user,
                                            password=self.pw,
                                            authSource='admin',
                                            authMechanism='SCRAM-SHA-1',
                                            overwrite=queued_exp['_id'])
        else:
            ex.observers[0] = MongoObserver(url=self.url,
                                            db_name=self.db_name,
                                            username=self.user,
                                            password=self.pw,
                                            authSource='admin',
                                            authMechanism='SCRAM-SHA-1')

        options = {'--force': True, '--name': queued_exp['experiment']['name']}
        if queued_exp['meta'].get('comment', None):
            options['--comment'] = queued_exp['meta']['comment']
        if queue:
            options['--queue'] = True
        elif 'fs_observer_path' in queued_exp['config']: # only check if not queue request
            options['--file_storage'] = queued_exp['config']['fs_observer_path']
            # ex.observers.append(FileStorageObserver(queued_exp['config']['fs_observer_path']))

        ex.run(config_updates=queued_exp['config'], options=options)
    
    # def _queue_post_process_experiment(self, pretrained_id): # TODO: pass additional config here


    def queue_experiment(self, mainfile, config, name=None):
        ex = self._get_sacred_exp(mainfile)

        options = {'--force': True,
            '--queue': True
            }
        if name:
            options['--name'] = name + '_' + ex.path

        ex.run(config_updates=config, options=options)
            
    def check_and_run_queue(self):
        logger = logging.getLogger('QAssistant::check_queue')
        
        queue_ids = self._get_queued_experiment_ids()
        logger.info(f'Queued IDs: {queue_ids}')

        if self.visible_exps:
            queue_ids = list(set(queue_ids) & set(self.visible_exps))
            logger.info(f'Filtered with visible Exp IDs: {queue_ids}')

        if not self.ignore_ready_check:
            queue_ids = self._filter_ready_experiment_ids(queue_ids)
        logger.info(f'Ready Exp IDs: {queue_ids}')

        free_gpus = self._get_free_gpus(max_used_percent=self.max_used_mem)
        logger.info(f'Free GPUs: {free_gpus}')

        gpu_uuids = [gpu['uuid'] for gpu in free_gpus]

        if len(free_gpus) > 0 and len(queue_ids) > 0:
            python_path = os.getenv('PYTHONPATH')
            for exp_id, gpu_id in zip(queue_ids[:len(free_gpus)], gpu_uuids):
                exp = self._load_experiment(exp_id)

                # fix missing comment field #hack
                exp['meta']['comment'] = exp['meta'].get('comment', None)

                root_dir = self._load_sources(exp_id)
                
                # create log files
                log_file_name = f'{str(exp_id).zfill(5)}_{exp["experiment"]["name"]}.log'
                log_file = os.path.join(self.logdir, log_file_name)
                
                cmd_list = ['nice', 'python3', '-c',
                            'from queue_assistant import start_experiment; start_experiment()',
                            '-exp_id', str(exp_id),
                            # '-gpu_id', str(gpu_id)
                            ]
            
                if 'env' in exp['config']:
                    conda_path = os.join(os.environ['CONDA_PATH'], 'bin/conda') if 'CONDA_PATH' in os.environ \
                        else os.path.join(os.environ['HOME'], 'miniconda3/bin/conda')
                    cmd_list = [conda_path, 'run','-n', exp['config']['env']] + cmd_list #TODO: find better way to specify path of conda
                    # cmd_list = ['conda', 'run','-n', exp['config']['env']] + cmd_list
                    
                logger.info('Starting Subprocess...')
                logger.info(' '.join(cmd_list))
                logger.info('Log File: %s' % log_file)
                with open(log_file, 'w+') as out_fd, open(os.devnull) as n_fd:
                    subprocess.Popen(cmd_list, 
                        stdout=out_fd, stderr=out_fd, stdin=n_fd, 
                        env={
                            'PYTHONPATH': root_dir,
                            'CUDA_VISIBLE_DEVICES': str(gpu_id), 
                            # 'CUDA_LAUNCH_BLOCKING': str(1), # otherwise, experiments fail undeterministically, probably because of pytorch nightly
                            'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'
                        }
                    )
        elif len(free_gpus) == 0 and len(queue_ids) > 0:
            logger.info('No GPU available, %i experiments pending...', len(queue_ids))
        elif len(free_gpus) > 0 and len(queue_ids) == 0:
            logger.info('Nothing to do (Queue empty)...')


def rerun_experiment():
    parser = argparse.ArgumentParser()

    parser.add_argument('--db_host', '-dbh', action='store', type=str,
                        default='soderbergh',
                        help='name of the sacred db host')
    parser.add_argument('--db_port', '-dbp', action='store', type=str,
                        default='27017')
    parser.add_argument('--db_name', '-dbn', action='store', type=str,
                        default='amatei_sacred')
    parser.add_argument('--visible_gpus', '-vgpu', action='store',
                        type=lambda str_list: str_list.split(','),
                        default='0')
    # parser.add_argument('--gpu_id', '-gpu_id', action='store', type=int, required=True)
    parser.add_argument('--exp_id', '-exp_id', action='store', type=int, required=True)
    args = parser.parse_args()

    assistant = QAssistant(port=args.db_port,
                           visible_gpus=args.visible_gpus
                           )

    exp = assistant._load_experiment(args.exp_id)
    assistant._start_experiment(exp,
                    # args.gpu_id,
                    replace=False)

def requeue_experiment():
    parser = argparse.ArgumentParser()

    parser.add_argument('--db_host', '-dbh', action='store', type=str,
                        default='soderbergh',
                        help='name of the sacred db host')
    parser.add_argument('--db_port', '-dbp', action='store', type=str,
                        default='27017')
    parser.add_argument('--db_name', '-dbn', action='store', type=str,
                        default='amatei_sacred')
    # parser.add_argument('--inplace', '-r', action='store_true')
    # parser.add_argument('--offset', '-o', action='store', type=int, default=100)
    parser.add_argument('--exp_id', '-exp_id', action='store', type=int, required=True)
    parser.add_argument('--exp_name', '-exp_name', action='store', type=str)
    # parser.add_argument('--update_source', '-u', action='store_true')
    parser.add_argument('--cfg_update', action='append', 
                        type=lambda kv: kv.split('='), dest='cfg_updates')
    args = parser.parse_args()

    if 'cfg_updates' in args:
        args.cfg_updates = dict(args.cfg_updates)

    if 'seed' in args.cfg_updates: # TODO: find better solution
        args.cfg_updates['seed'] = int(args.cfg_updates['seed'])

    logger = logging.getLogger('QAssistant::requeue_experiment')

    assistant = QAssistant(
                db_host=args.db_host,
                port=args.db_port, 
                db_name=args.db_name,
            )

    logger.info('Obtaining run from DB')
    exp = assistant._load_experiment(args.exp_id)
    # fs_observer_path = Path(run['config']['fs_observer_path']) / str(run['_id'])

    if 'exp_name' in args:
        exp['experiment']['name'] = args.exp_name   

    if args.cfg_updates:
        logger.info(f'Updating config with {args.cfg_updates}')
        exp['config'].update(args.cfg_updates)
    else:
        logger.info('No config updates')

    exp['config']['rerun_of'] = args.exp_id

    assistant._start_experiment(exp, queue=True, replace=False)

    # tmp_root = assistant._load_sources(args.exp_id)

    # if args.update_source:
    #     logger.info('Using latest source code')
    #     root_dir = os.getcwd()
    # else: 
    #     logger.info('Using source code from DB')
    #     root_dir = tmp_root

    # logger.info(f'Root dir: {root_dir}')
    
    # cmd_list = ['/usr/bin/python3', run['experiment']['mainfile'] , 'with',
    #             tmp_root + '/config.yaml',
    #             '-n', exp['experiment']['name'],
    #             '-f',
    #             '-q'
    #         ]

    # if args.inplace:
    #     # find highest _id + 1 + offset
    #     new_id = assistant.db.runs.find_one(sort=[('_id', -1)])['_id'] + 1 + args.offset
    #     logger.info(f'Moving experiment {run["_id"]} -> {new_id}')
    #     run['experiment']['name'] += f' was _id={run["_id"]}'
    #     run['_id'] = new_id

    #     #TODO: also correct run_id fields in metrics collection
        
    #     # move old run 
    #     assistant.db.runs.insert_one(run)
    #     assistant.db.runs.delete_one({'_id': args.exp_id})

    #     cmd_list += ['-i', str(args.exp_id)]

    #     # move old _id experiment to _id_<start_time>
    #     if fs_observer_path.is_dir(): # if exists
    #         new_path = f"{str(fs_observer_path)}_{str(run['start_time'])}" 
    #         logger.info(f'Moving {fs_observer_path} -> {new_path}')
    #         fs_observer_path.rename(new_path)
    # else:
    #    pass 

    # logger.info('Constructed command')
    # logger.info(cmd_list)

    # subprocess.Popen(cmd_list, cwd=root_dir,
    #     env={
    #         'PYTHONPATH': root_dir,
    #     }
    # )

def start_experiment():
    # argument parsing
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--db_host', '-dbh', action='store', type=str,
                        default='soderbergh',
                        help='name of the sacred db host')
    parser.add_argument('--db_port', '-dbp', action='store', type=str,
                        default='27017')
    parser.add_argument('--db_name', '-dbn', action='store', type=str,
                        default='amatei_sacred')
    parser.add_argument('--visible_gpus', '-vgpu', action='store',
                        type=lambda str_list: str_list.split(','),
                        default='0')
    # parser.add_argument('--gpu_id', '-gpu_id', action='store', required=True)
    parser.add_argument('--exp_id', '-exp_id', action='store', type=int, required=True)
    args = parser.parse_args()
    
    assistant = QAssistant(port=args.db_port, 
                           visible_gpus=args.visible_gpus)
    
    exp = assistant._load_experiment(args.exp_id)
    assistant._start_experiment(exp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--visible_gpus', '-vgpu', action='store',
                        type=lambda str_list: str_list.split(','),
                        default='0')
    parser.add_argument('--visible_exps', '-vexps', action='store',
                        type=lambda str_list: [int(s) for s in str_list.split(',')],
                        default=None)
    parser.add_argument('--max_used_mem_perc', '-mem', action='store',
                        type=int, default=3)
    parser.add_argument('--ignore_ready_check', '-irc', action='store_true')
    parser.add_argument('--filter_by_hostname', type=bool, default=True)
    args = parser.parse_args()

    assistant = QAssistant(visible_gpus=args.visible_gpus,
                           visible_exps=args.visible_exps,
                           max_used_mem=args.max_used_mem_perc,
                           ignore_ready_check=args.ignore_ready_check,
                           filter_by_hostname=args.filter_by_hostname
                           )
    assistant.check_and_run_queue()
