'''
Created on May 17, 2019

@author: fmoya
'''

from __future__ import print_function
import os
import io
import logging
import torch
import numpy as np
import random

import platform
from modus_selecter import Modus_Selecter

import datetime
from pathlib import Path

from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver

from queue_assistant import load_credentials, configure_sacred

configure_sacred()

now = datetime.datetime.now()
ex= Experiment('ICPR 2024')

if os.uname()[1] in ['rosenblatt', 'cameron']:
    user, pw, url, db_name = load_credentials(path='~/.mongodb_credentials')
    ex.observers.append(MongoObserver.create(url=url,
                                            db_name=db_name,
                                            username=user,
                                            password=pw,
                                            authSource='admin',
                                            authMechanism='SCRAM-SHA-1'))



def setup_experiment_logger(logging_level=logging.DEBUG, filename=None):
    # set up the logging
    logging_format = '[%(asctime)-19s, %(name)s, %(levelname)s] %(message)s'

    if filename != None:
        logging.basicConfig(filename=filename, level=logging.DEBUG,
                            format=logging_format,
                            filemode='w')
    else:
        logging.basicConfig(level=logging_level,
                            format=logging_format,
                            filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger

    if logging.getLogger('').hasHandlers():
        logging.getLogger('').handlers.clear()

    logging.getLogger('').addHandler(console)

    return



@ex.config
def my_config():
    print("configuration function began")
    
    dataset = 'mocap'
    dataset_finetuning = dataset
    network = 'cnn'
    output = 'softmax'
    reshape_input = False
    usage_modus = 'train'

    name_counter = 0
    sacred = True

    #dataset_finetuning = config["dataset_finetuning"]
    #pooling = config["pooling"]

    assert dataset in ['mocap', 'mbientlab', 'mobiact', 'motionsense', 'sisfall'], 'Dataset is configured wrong'
    assert network in ['cnn', 'lstm', 'cnn_imu', 'cnn_transformer'], 'Network is configured wrong'
    assert output in ['softmax', 'attribute'], 'Output is configured wrong'
    assert usage_modus in ['train', 'test',  'evolution',  'train_final',  'train_random',  'fine_tuning'], 'usage_mouds configured wrong'

    lr = 0.001
    seed=42

    # Number of repeated trainings
    repetitions = 1
    
    # Flags
    plotting = False

    # Options

    dataset_root_defaults = {
        'mocap':        "/vol/actrec/DFG_Project/2019/LARa_dataset/MoCap/LARa_dataset_mocap/",
        'mbientlab':    "/vol/actrec/DFG_Project/2019/LARa_dataset/Mbientlab/LARa_dataset_mbientlab/",
        'mobiact':      "/vol/actrec/MobiAct_Dataset/",
        'motionsense':  "/vol/actrec/motion-sense-master/data/A_DeviceMotion_data/A_DeviceMotion_data",
        'sisfall':      "/vol/actrec/SisFall_dataset"
        }
    dataset_root = dataset_root_defaults[dataset]

    # Dataset Hyperparameters
    NB_sensor_channels_defaults = {'mocap': 126, 'mbientlab': 30, 'mobiact': 9, 'motionsense': 9, 'sisfall': 9}
    sliding_window_length_defaults = {'mocap': 200, 'mbientlab': 100, 'mobiact': 200, 'motionsense': 200, 'sisfall': 200}
    sliding_window_step_defaults = {'mocap': 25, 'mbientlab': 12, 'mobiact': 50, 'motionsense': 25, 'sisfall': 50}

    NB_sensor_channels = NB_sensor_channels_defaults[dataset]
    sliding_window_length = sliding_window_length_defaults[dataset]
    sliding_window_step = sliding_window_step_defaults[dataset]
    
    # Number of classes for either for activity recognition
    num_classes_defaults = {'mocap': 7, 'mbientlab': 7, 'mobiact': 9, 'motionsense': 6, 'sisfall': 15}
    num_attributes_defaults = {'mocap': 19, 'mbientlab': 19, 'mobiact': 0, 'motionsense': 0, 'sisfall': 0}
    num_tr_inputs_defaults = {'mocap': 345417, 'mbientlab': 94753, 'mobiact': 160561, 'motionsense': 118671, 'sisfall': 118610}

    num_classes = num_classes_defaults[dataset]
    num_attributes = num_attributes_defaults[dataset]
    num_tr_inputs = num_tr_inputs_defaults[dataset]


    # It was thought to have different LR per dataset, but experimentally have worked the next three
    # Learning rate
    # learning_rates = [0.001, 0.0001, 0.00001, 0.000001]
    # lr = {'mocap': {'cnn': learning_rates[learning_rates_idx],
    #                 'lstm': learning_rates[learning_rates_idx],
    #                 'cnn_imu': learning_rates[learning_rates_idx],
    #                 'cnn_transformer':learning_rates[learning_rates_idx]},
    #       'mbientlab': {'cnn': learning_rates[learning_rates_idx],
    #                     'lstm': learning_rates[learning_rates_idx],
    #                     'cnn_imu': learning_rates[learning_rates_idx],
    #                     'cnn_transformer': learning_rates[learning_rates_idx]},
    #       'mobiact': {'cnn': learning_rates[learning_rates_idx],
    #                   'lstm': learning_rates[learning_rates_idx],
    #                   'cnn_imu': learning_rates[learning_rates_idx],
    #                   'cnn_transformer':learning_rates[learning_rates_idx]},
    #       'motionsense': {'cnn': learning_rates[learning_rates_idx],
    #                      'lstm': learning_rates[learning_rates_idx],
    #                      'cnn_imu': learning_rates[learning_rates_idx],
    #                      'cnn_transformer':learning_rates[learning_rates_idx]},
    #       'sisfall': {'cnn': learning_rates[learning_rates_idx],
    #                   'lstm': learning_rates[learning_rates_idx],
    #                   'cnn_imu': learning_rates[learning_rates_idx],
    #                   'cnn_transformer':learning_rates[learning_rates_idx]}
          
    #       }
    lr_mult = 1.0

    # Maxout
    use_maxout_defaults = {'cnn': False, 'lstm': False, 'cnn_imu': False, 'cnn_transformer':False}
    use_maxout = use_maxout_defaults[network]

    # Balacing the proportion of classes into the dataset dataset
    # This will be deprecated
    balancing_defaults = {'mocap': False, 'mbientlab': False, 'mobiact': False, 'motionsense': False, 'sisfall': False}
    balancing = balancing_defaults[dataset]

    # Epochs
    if usage_modus == 'train_final' or usage_modus == 'fine_tuning':
        epoch_mult = 2
    else:
        epoch_mult = 1

    # Number of epochs depending of the dataset and network
    

    epochs_defaults =  \
            {'mocap': {'cnn': {'softmax': 6, 'attribute': 6},
                        'lstm': {'softmax': 15, 'attribute': 6},
                        'cnn_imu': {'softmax': 10, 'attribute': 6},
                        'cnn_transformer':{'softmax': 15, 'attribute': 6}},
              'mbientlab': {'cnn': {'softmax': 10, 'attribute': 10},
                            'lstm': {'softmax': 15, 'attribute': 10},
                            'cnn_imu': {'softmax': 30, 'attribute': 10},
                            'cnn_transformer':{'softmax': 50, 'attribute': 6}},
              'mobiact': {'cnn': {'softmax': 10, 'attribute': 50},
                          'lstm': {'softmax': 15, 'attribute': 5},
                          'cnn_imu': {'softmax': 32, 'attribute': 50},
                          'cnn_transformer':{'softmax': 15, 'attribute': 6}},
              'motionsense': {'cnn': {'softmax': 30, 'attribute': 50},
                             'lstm': {'softmax': 15, 'attribute': 5},
                             'cnn_imu': {'softmax': 32, 'attribute': 10},
                             'cnn_transformer':{'softmax': 15, 'attribute': 6}},
              'sisfall': {'cnn': {'softmax': 30, 'attribute': 50},
                                  'lstm': {'softmax': 15, 'attribute': 5},
                                  'cnn_imu': {'softmax': 32, 'attribute': 50},
                                  'cnn_transformer':{'softmax': 15, 'attribute': 6}},
              
              }
    epochs = epochs_defaults[dataset][network][output]

    augmentations =  'none'
    assert augmentations in ['none',  'time_warp',  'time_warp_seed',  'jittering',  'scaling',  'flipping',  'magnitude_warping',
                      'permutation',  'slicing',  'window_warping', 'tilt', 'spawner'], 'augmentation configured wrong'
    
    division_epochs_defaults = {'mocap': 2, 'mbientlab': 1, 'mobiact': 1, 'motionsense': 1, 'sisfall': 1}
    division_epochs = division_epochs_defaults[dataset]

    # Batch size
    batch_size_train_defaults = {
        'cnn': {'mocap': 100, 'mbientlab': 100, 'mobiact': 100, 'motionsense': 50, 'sisfall': 50},
        'lstm': {'mocap': 50, 'mbientlab': 50, 'mobiact': 100, 'motionsense': 50, 'sisfall': 50},
        'cnn_imu': {'mocap': 100, 'mbientlab': 100, 'mobiact': 100, 'motionsense': 100, 'sisfall': 100},
        'cnn_transformer': {'mocap': 50, 'mbientlab': 128, 'mobiact': 200, 'motionsense': 50, 'sisfall': 50}}

    batch_size_val_defaults = {'cnn': {'mocap': 100, 'mbientlab': 100, 'mobiact': 100, 'motionsense': 50,'sisfall': 50},
                      'lstm': {'mocap': 50, 'mbientlab': 50, 'mobiact': 100, 'motionsense': 50,'sisfall': 50},
                      'cnn_imu': {'mocap': 100, 'mbientlab': 100,'mobiact': 100, 'motionsense': 100,'sisfall': 100},
                      'cnn_transformer': {'mocap': 50, 'mbientlab': 128,'mobiact': 200, 'motionsense': 50,'sisfall': 50}}

    batch_size_train = batch_size_train_defaults[network][dataset]
    batch_size_val = batch_size_val_defaults[network][dataset]
    

    # Number of iterations for accumulating the gradients
    accumulation_steps_defaults = {'mocap': 4, 'mbientlab': 4, 'mobiact': 4, 'motionsense': 4, 'sisfall': 4}
    accumulation_steps = accumulation_steps_defaults[dataset]

    # Filters
    filter_size_defaults = {'mocap': 5, 'mbientlab': 5, 'mobiact': 5, 'motionsense': 5, 'sisfall': 5}
    filter_size = filter_size_defaults[dataset]
    num_filters_defaults = {'mocap': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64, 'cnn_transformer':64},
                   'mbientlab': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64, 'cnn_transformer':64},
                   'mobiact': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64, 'cnn_transformer':64},
                   'motionsense': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64, 'cnn_transformer':64},
                   'sisfall': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64, 'cnn_transformer':64}
                   }
    num_filters = num_filters_defaults[dataset][network]
    
    #lstm hyperparameters hidden layer and layer dimension
    lstm_hidlyr= 256
    hidden_layer = lstm_hidlyr
    lstm_lyrdim= 4
    layer_dim = lstm_lyrdim
    
    #cnntransformer hyperparameters
    transformer_dim=64
    n_head=8
    dim_fc=128
    n_layers=6
    n_embedding_layers=4
    use_pos_embedding=True

    transformer_heads = n_head
    transformer_fc = dim_fc
    transformer_layers = n_layers
    trans_pos_embed = use_pos_embedding
    trans_embed_layer = n_embedding_layers

    freeze_options = False
    assert freeze_options in [True, False], 'ConfigurationError: freeze_options'

    # Evolution
    evolution_iter = 10000

    # Results and network will be stored in different folders according to the dataset and network
    # Network that will be trained and tested are stored in these folders
    # This as a sort of organisation for tracking the experiments
    # dataset/network/output/MLP_type/input_shape/
    # dataset/network/output/MLP_type/input_shape/experiment
    # dataset/network/output/MLP_type/input_shape/experiment/plots
    # dataset/network/output/MLP_type/input_shape/final
    # dataset/network/output/MLP_type/input_shape/final/plots
    # dataset/network/output/MLP_type/input_shape/fine_tuning
    # dataset/network/output/MLP_type/input_shape/fine_tuning/plots

    if reshape_input:
        reshape_folder = "reshape"
    else:
        reshape_folder = "noreshape"

    # if fully_convolutional:
    #     fully_convolutional = "FCN"
    # else:
    #     fully_convolutional = "FC"

    fully_convolutional = "FC"
    assert fully_convolutional in ["FC", "FCN"], "ConfigurationError: fully_convolutional"

    run_id = None
    base_folder_exp = '/data/nnair/icpr2024/'
    folder_exp = None  # don't set this by hand, set base_folder_exp instead

    if output == 'softmax':
        labeltype = "class"
        folder_exp_defaults = { 
            'mocap':        str(Path(base_folder_exp) / "lara/results/"),
            'mbientlab':    str(Path(base_folder_exp) / "lara_imu/results/"),
            'mobiact':      str(Path(base_folder_exp) / "mobiact/results/"),
            'motionsense':  str(Path(base_folder_exp) / "motionsense/results/"),
            'sisfall':      str(Path(base_folder_exp) / "sisfall/results/")
                    }
        if folder_exp is None: # if not set via cmd pick from defaults
            folder_exp =  _check_and_create_fldr(folder_exp_defaults[dataset])
        if run_id is None: # if not set by cmd pick highest
            run_id = _maximum_existing_run_id(folder_exp) + 1
        folder_exp = str(_check_and_create_fldr(Path(folder_exp) / str(run_id)))

    elif output == 'attribute':
        labeltype = "attributes"
        folder_exp = "/data/nnair/idnetwork/results/all/"


    # GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5" # set via shell
    GPU = 0 # first in list of CUDA_VISIBILE_DEVICES

    # Labels position on the segmented window
    label_pos = 'end'
    assert label_pos in  ['middle', 'mode', 'end'], f'ConifigurationError: label_pos'

    percentages_names = ["001", "002", "005", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    percentages_dataset = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    #TODO not needed anymore; log_per_epoch handles the same
    #train_show_value = num_tr_inputs[dataset[dataset_idx]] * percentages_dataset[percentage_idx]
    train_show_value = num_tr_inputs / batch_size_train
    if dataset == "mbientlab" or dataset == "motionminers_real":
        train_show_defaults = {'cnn': int(train_show_value / 50), 'lstm': 50, 'cnn_imu': int(train_show_value / 50), 'cnn_transformer':int(train_show_value / 50)}
        valid_show_defaults = {'cnn': int(train_show_value / 10), 'lstm': 10, 'cnn_imu': int(train_show_value / 10), 'cnn_transformer': int(train_show_value / 10)}
        train_show = train_show_defaults[network]
        valid_show = valid_show_defaults[network]
    elif dataset == "mocap":
        train_show_defaults = {'cnn': int(train_show_value / 100), 'lstm': 100, 'cnn_imu': int(train_show_value / 100), 'cnn_transformer': int(train_show_value / 100)}
        valid_show_defaults = {'cnn': int(train_show_value / 20), 'lstm': 50, 'cnn_imu': int(train_show_value / 20), 'cnn_transformer':int(train_show_value / 20)}
        train_show = train_show_defaults[network]
        valid_show = valid_show_defaults[network]
    else:
        train_show_defaults = {'cnn': int(train_show_value / 100), 'lstm': 100, 'cnn_imu': int(train_show_value / 100), 'cnn_transformer': int(train_show_value / 100)}
        valid_show_defaults = {'cnn': int(train_show_value / 50), 'lstm': 50, 'cnn_imu': int(train_show_value / 50), 'cnn_transformer': int(train_show_value / 50)}
        train_show = train_show_defaults[network]
        valid_show = valid_show_defaults[network]

    log_per_epoch = 20

    file_suffix =  'results_yy{}mm{}dd{:02d}hh{:02d}mm{:02d}.xml'.format(now.year,
                                                                        now.month,
                                                                        now.day,
                                                                        now.hour,
                                                                        now.minute)



'''
    print("configuration function began")
    dataset_idx = [0]
    network_idx = [2]
    reshape_input = [False]
    #dataset_ft_idx = [0,1,2,3]
    counter_exp = 0
    freeze = [0]
    percentages = [12]
    output_idxs = [0]
    lrs = [0]
    for dts in range(len(dataset_idx)):
        for nt in range(len(network_idx)):
            for opt in output_idxs:
                #for dft in dataset_ft_idx:
                    for pr in percentages:
                        for rsi in range(len(reshape_input)):
                            for fr in freeze:
                                for lr in lrs:
                                    config = configuration(dataset_idx=dataset_idx[dts],
                                                           network_idx=network_idx[nt],
                                                           output_idx=opt,
                                                           usage_modus_idx=0,
                                                           #dataset_fine_tuning_idx=dft,
                                                           reshape_input=reshape_input[rsi],
                                                           learning_rates_idx=lr,
                                                           name_counter=counter_exp,
                                                           freeze=fr,
                                                           percentage_idx=pr,
                                                           fully_convolutional=False
                                                           )

    
    dataset = config["dataset"]
    network = config["network"]
    output = config["output"]
    reshape_input = config["reshape_input"]
    usageModus = config["usage_modus"]
    lr = config["lr"]
    bsize = config["batch_size_train"]
    dataet_root= config["dataset_root"]
    '''
   

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Torch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Python RNG
    np.random.seed(seed)
    random.seed(seed)

def _maximum_existing_run_id(basedir):
    dir_nrs = [int(d) for d in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, d)) and d.isdigit()]
    if dir_nrs:
        return max(dir_nrs)
    else:
        return 0

def _check_and_create_fldr(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder)


# @ex.capture
# def run(config, dataset, network, output, usage_modus):
@ex.automain
def run(_config):
    config = _config
    seed_everything(config['seed'])
    file_name='/data/nnair/icpr2024/'
   
    file_name='/data/nnair/icpr2024/'+'logger.txt'
    
    setup_experiment_logger(logging_level=logging.DEBUG,filename=file_name)

    logging.info('Finished')
    logging.info('Dataset {} Network {} Output {} Modus {}'.format(config['dataset'], config['network'], config['output'], config['usage_modus']))

    modus = Modus_Selecter(config, ex)

    # Starting process
    modus.net_modus()
    
    print("Done")

# @ex.automain
# def main(config):
#     print("main began")
#     #Setting the same RNG seed

#     print("Python  {}".format(platform.python_version()))


#     run(config)

#     print("Done")

'''
def main():
    """
    Run experiment for a certain set of parameters

    User is welcome to revise in detatil the configuration function
    for more information about all of possible configurations for the experiments

    """
    dataset_idx = [3]
    network_idx = [2]
    reshape_input = [False]
    #dataset_ft_idx = [0,1,2,3]
    counter_exp = 0
    freeze = [0]
    percentages = [12]
    output_idxs = [0]
    lrs = [0]
    for dts in range(len(dataset_idx)):
        for nt in range(len(network_idx)):
            for opt in output_idxs:
                #for dft in dataset_ft_idx:
                    for pr in percentages:
                        for rsi in range(len(reshape_input)):
                            for fr in freeze:
                                for lr in lrs:
                                    config = configuration(dataset_idx=dataset_idx[dts],
                                                           network_idx=network_idx[nt],
                                                           output_idx=opt,
                                                           usage_modus_idx=0,
                                                           #dataset_fine_tuning_idx=dft,
                                                           reshape_input=reshape_input[rsi],
                                                           learning_rates_idx=lr,
                                                           name_counter=counter_exp,
                                                           freeze=fr,
                                                           percentage_idx=pr,
                                                           fully_convolutional=False)

                                    setup_experiment_logger(logging_level=logging.DEBUG,
                                                            filename=config['folder_exp'] + "logger.txt")

                                    logging.info('Finished')

                                    modus = Modus_Selecter(config)

                                    # Starting process
                                    modus.net_modus()
                                    counter_exp += 1


    return



if __name__ == '__main__':

    #Setting the same RNG seed
    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Torch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Python RNG
    np.random.seed(seed)
    random.seed(seed)

    print("Python Platform {}".format(platform.python_version()))
    
    main()

    print("Done")
'''

