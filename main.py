'''
Created on May 17, 2019

@author: fmoya
'''

from __future__ import print_function
import os
import logging
import torch
import numpy as np
import random

import platform
from modus_selecter import Modus_Selecter

import datetime

from sacred import Experiment
from sacred.observers import MongoObserver

ex= Experiment('mobiact lstm x3 50-0.001-15 trial1')

ex.observers.append(MongoObserver.create(url='curtiz',
                                         db_name='nnair_sacred',
                                         username='nnair',
                                         password='Germany2018',
                                         authSource='admin',
                                         authMechanism='SCRAM-SHA-1'))

def configuration(dataset_idx, network_idx, output_idx, usage_modus_idx=0, dataset_fine_tuning_idx=0,
                  reshape_input=False, learning_rates_idx=0, name_counter=0, freeze=0, percentage_idx=0,
                  fully_convolutional=False, sacred =True):
    """
    Set a configuration of all the possible variables that were set in the experiments.
    This includes the datasets, hyperparameters for training, networks, outputs, datasets paths,
    results paths

    @param dataset_idx: id of dataset
    @param network_idx: id of network 0 for tcnn, 1, for tcnn-lstm, 2 tcnn-IMU
    @param output_idx: 0 for softmax, 1 for attributes
    @param usage_modus_idx: id of the modus 0 for train, 1 for test, 2 for evolution, 3 for train_final,...
    @param dataset_fine_tuning_idx: id of source dataset in case of finetuning
    @param reshape_input: reshaping the input False for [C,T] or, True for [3,C/3,T]=[[x,y,z], [sensors], Time]
    @param learning_rates_idx: id for the Learning Rate
    @param name_counter: counter for experiments
    @param name_counter: 0 for freezing the CNN layers, or 1 for fine-tuning
    @param percentage_idx: Percentage for the training dataset
    @param fully_convolutional: False for FC or True for FCN
    @return configuration: dict with all the configurations
    """
    # Flags
    plotting = False

    # Options
    dataset = {0: 'mocap', 1: 'mbientlab', 2: 'mobiact', 3: 'motionsense', 4: 'sisfall' }
    network = {0: 'cnn', 1: 'lstm', 2: 'cnn_imu', 3: 'cnn_transformer'}
    output = {0: 'softmax', 1: 'attribute'}
    usage_modus = {0: 'train', 1: 'test', 2: 'evolution', 3: 'train_final', 4: 'train_random', 5: 'fine_tuning'}

    # Dataset Hyperparameters
    NB_sensor_channels = {'mocap': 126, 'mbientlab': 30, 'mobiact': 9, 'motionsense': 9, 'sisfall': 9}
    sliding_window_length = {'mocap': 200, 'mbientlab': 100, 'mobiact': 200, 'motionsense': 200, 'sisfall': 200}
    sliding_window_step = {'mocap': 25, 'mbientlab': 12, 'mobiact': 50, 'motionsense': 25, 'sisfall': 50}
    
    # Number of classes for either for activity recognition
    num_classes = {'mocap': 7, 'mbientlab': 7, 'mobiact': 9, 'motionsense': 6, 'sisfall': 15}
    num_attributes = {'mocap': 19, 'mbientlab': 19, 'mobiact': 0, 'motionsense': 0, 'sisfall': 0}
    num_tr_inputs = {'mocap': 345417, 'mbientlab': 94753, 'mobiact': 160561, 'motionsense': 118671, 'sisfall': 118610}

    


    # It was thought to have different LR per dataset, but experimentally have worked the next three
    # Learning rate
    learning_rates = [0.001, 0.0001, 0.00001, 0.000001]
    lr = {'mocap': {'cnn': learning_rates[learning_rates_idx],
                    'lstm': learning_rates[learning_rates_idx],
                    'cnn_imu': learning_rates[learning_rates_idx],
                    'cnn_transformer':learning_rates[learning_rates_idx]},
          'mbientlab': {'cnn': learning_rates[learning_rates_idx],
                        'lstm': learning_rates[learning_rates_idx],
                        'cnn_imu': learning_rates[learning_rates_idx],
                        'cnn_transformer': learning_rates[learning_rates_idx]},
          'mobiact': {'cnn': learning_rates[learning_rates_idx],
                      'lstm': learning_rates[learning_rates_idx],
                      'cnn_imu': learning_rates[learning_rates_idx],
                      'cnn_transformer':learning_rates[learning_rates_idx]},
          'motionsense': {'cnn': learning_rates[learning_rates_idx],
                         'lstm': learning_rates[learning_rates_idx],
                         'cnn_imu': learning_rates[learning_rates_idx],
                         'cnn_transformer':learning_rates[learning_rates_idx]},
          'sisfall': {'cnn': learning_rates[learning_rates_idx],
                      'lstm': learning_rates[learning_rates_idx],
                      'cnn_imu': learning_rates[learning_rates_idx],
                      'cnn_transformer':learning_rates[learning_rates_idx]}
          
          }
    lr_mult = 1.0

    # Maxout
    use_maxout = {'cnn': False, 'lstm': False, 'cnn_imu': False, 'cnn_transformer':False}

    # Balacing the proportion of classes into the dataset dataset
    # This will be deprecated
    balancing = {'mocap': False, 'mbientlab': False, 'mobiact': False, 'motionsense': False, 'sisfall': False}

    # Epochs
    if usage_modus[usage_modus_idx] == 'train_final' or usage_modus[usage_modus_idx] == 'fine_tuning':
        epoch_mult = 2
    else:
        epoch_mult = 1

    # Number of epochs depending of the dataset and network
    epochs = {'mocap': {'cnn': {'softmax': 6, 'attribute': 6},
                        'lstm': {'softmax': 15, 'attribute': 6},
                        'cnn_imu': {'softmax': 10, 'attribute': 6},
                        'cnn_transformer':{'softmax': 50, 'attribute': 6}},
              'mbientlab': {'cnn': {'softmax': 10, 'attribute': 10},
                            'lstm': {'softmax': 15, 'attribute': 10},
                            'cnn_imu': {'softmax': 30, 'attribute': 10},
                            'cnn_transformer':{'softmax': 50, 'attribute': 6}},
              'mobiact': {'cnn': {'softmax': 10, 'attribute': 50},
                          'lstm': {'softmax': 15, 'attribute': 5},
                          'cnn_imu': {'softmax': 32, 'attribute': 50},
                          'cnn_transformer':{'softmax': 15, 'attribute': 6}},
              'motionsense': {'cnn': {'softmax': 30, 'attribute': 50},
                             'lstm': {'softmax': 30, 'attribute': 5},
                             'cnn_imu': {'softmax': 32, 'attribute': 10},
                             'cnn_transformer':{'softmax': 15, 'attribute': 6}},
              'sisfall': {'cnn': {'softmax': 50, 'attribute': 50},
                                  'lstm': {'softmax': 15, 'attribute': 5},
                                  'cnn_imu': {'softmax': 32, 'attribute': 50},
                                  'cnn_transformer':{'softmax': 30, 'attribute': 6}},
              
              }

    augmentations = {0: 'none', 1: 'time_warp', 2: 'time_warp_seed', 3: 'jittering', 4: 'scaling', 5: 'flipping', 6: 'magnitude_warping',
                     7: 'permutation', 8: 'slicing', 9: 'window_warping', 10: 'tilt', 11: 'spawner'}
    
    division_epochs = {'mocap': 2, 'mbientlab': 1, 'mobiact': 1, 'motionsense': 1, 'sisfall': 1}

    # Batch size
    batch_size_train = {
        'cnn': {'mocap': 100, 'mbientlab': 100, 'mobiact': 100, 'motionsense': 50, 'sisfall': 50},
        'lstm': {'mocap': 50, 'mbientlab': 50, 'mobiact': 50, 'motionsense': 100, 'sisfall': 50},
        'cnn_imu': {'mocap': 100, 'mbientlab': 100, 'mobiact': 100, 'motionsense': 100, 'sisfall': 100},
        'cnn_transformer': {'mocap': 100, 'mbientlab': 128, 'mobiact': 50, 'motionsense': 100, 'sisfall': 50}}

    batch_size_val = {'cnn': {'mocap': 100, 'mbientlab': 100, 'mobiact': 100, 'motionsense': 50,'sisfall': 50},
                      'lstm': {'mocap': 50, 'mbientlab': 50, 'mobiact': 50, 'motionsense': 100,'sisfall': 50},
                      'cnn_imu': {'mocap': 100, 'mbientlab': 100,'mobiact': 100, 'motionsense': 100,'sisfall': 100},
                      'cnn_transformer': {'mocap': 100, 'mbientlab': 128,'mobiact': 50, 'motionsense': 100,'sisfall': 50}}

    # Number of iterations for accumulating the gradients
    accumulation_steps = {'mocap': 4, 'mbientlab': 4, 'mobiact': 4, 'motionsense': 4, 'sisfall': 4}

    # Filters
    filter_size = {'mocap': 5, 'mbientlab': 5, 'mobiact': 5, 'motionsense': 5, 'sisfall': 5}
    num_filters = {'mocap': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64, 'cnn_transformer':64},
                   'mbientlab': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64, 'cnn_transformer':64},
                   'mobiact': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64, 'cnn_transformer':64},
                   'motionsense': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64, 'cnn_transformer':64},
                   'sisfall': {'cnn': 64, 'lstm': 64, 'cnn_imu': 64, 'cnn_transformer':64}
                   }
    
    #lstm hyperparameters hidden layer and layer dimension
    lstm_hidlyr= 256
    lstm_lyrdim= 4
    
    #cnntransformer hyperparameters
    transformer_dim=64
    n_head=8
    dim_fc=128
    n_layers=6
    n_embedding_layers=4
    use_pos_embedding=True

    freeze_options = [False, True]

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

    reshape_input = reshape_input
    if reshape_input:
        reshape_folder = "reshape"
    else:
        reshape_folder = "noreshape"

    if fully_convolutional:
        fully_convolutional = "FCN"
    else:
        fully_convolutional = "FC"

    # User gotta take care of creating these folders, or storing the results in a different way
    
    if output[output_idx] == 'softmax':
        labeltype = "class"
        folder_exp = {'mocap': "/data/nnair/icpr2024/lara/results/transt/",
                    'mbientlab': "/data/nnair/icpr2024/lara_imu/results/transt/",
                    'mobiact': "/data/nnair/icpr2024/mobiact/results/trial1/",
                    'motionsense': "/data/nnair/icpr2024/motionsense/results/trial/",
                    'sisfall': "/data/nnair/icpr2024/sisfall/results/trial/"
                    }
    elif output[output_idx] == 'attribute':
        labeltype = "attributes"
        folder_exp = "/data/nnair/idnetwork/results/all/"


    
    # Paths are given according to the ones created in *preprocessing.py for the datasets
    
    dataset_root = {'mocap': "/data/nnair/icpr2024/lara/prepros/",
                    'mbientlab': "/data/nnair/icpr2024/lara_imu/prepros/",
                    'mobiact': "/data/nnair/icpr2024/mobiact/prepros/",
                    'motionsense': "/data/nnair/icpr2024/motionsense/prepros/",
                    'sisfall': "/data/nnair/icpr2024/sisfall/prepros/"
                    }

    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    GPU = 0

    # Labels position on the segmented window
    label_pos = {0: 'middle', 1: 'mode', 2: 'end'}

    percentages_names = ["001", "002", "005", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    percentages_dataset = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    #train_show_value = num_tr_inputs[dataset[dataset_idx]] * percentages_dataset[percentage_idx]
    train_show_value = num_tr_inputs[dataset[dataset_idx]] / \
                       batch_size_train[network[network_idx]][dataset[dataset_idx]]
    if dataset[dataset_idx] == "mbientlab" or dataset[dataset_idx] == "motionminers_real":
        train_show = {'cnn': int(train_show_value / 50), 'lstm': 50, 'cnn_imu': int(train_show_value / 50), 'cnn_transformer':int(train_show_value / 50)}
        valid_show = {'cnn': int(train_show_value / 10), 'lstm': 10, 'cnn_imu': int(train_show_value / 10), 'cnn_transformer': int(train_show_value / 10)}
    elif dataset[dataset_idx] == "mocap":
        train_show = {'cnn': int(train_show_value / 100), 'lstm': 100, 'cnn_imu': int(train_show_value / 100), 'cnn_transformer': int(train_show_value / 100)}
        valid_show = {'cnn': int(train_show_value / 20), 'lstm': 50, 'cnn_imu': int(train_show_value / 20), 'cnn_transformer':int(train_show_value / 20)}
    else:
        train_show = {'cnn': int(train_show_value / 100), 'lstm': 100, 'cnn_imu': int(train_show_value / 100), 'cnn_transformer': int(train_show_value / 100)}
        valid_show = {'cnn': int(train_show_value / 50), 'lstm': 50, 'cnn_imu': int(train_show_value / 50), 'cnn_transformer': int(train_show_value / 50)}

    now = datetime.datetime.now()


    configuration = {'dataset': dataset[dataset_idx],
                     'dataset_finetuning': dataset[dataset_fine_tuning_idx],
                     'network': network[network_idx],
                     'output': output[output_idx],
                     'num_filters': num_filters[dataset[dataset_idx]][network[network_idx]],
                     'filter_size': filter_size[dataset[dataset_idx]],
                     'lr': lr[dataset[dataset_idx]][network[network_idx]] * lr_mult,
                     'epochs': epochs[dataset[dataset_idx]][network[network_idx]][output[output_idx]] * epoch_mult,
                     'evolution_iter': evolution_iter,
                     'train_show': train_show[network[network_idx]],
                     'valid_show': valid_show[network[network_idx]],
                     'plotting': plotting,
                     'usage_modus': usage_modus[usage_modus_idx],
                     'folder_exp': folder_exp[dataset[dataset_idx]],
                     #'folder_exp_base_fine_tuning': folder_exp_base_fine_tuning,
                     'use_maxout': use_maxout[network[network_idx]],
                     'balancing': balancing[dataset[dataset_idx]],
                     'GPU': GPU,
                     'division_epochs': division_epochs[dataset[dataset_idx]],
                     'NB_sensor_channels': NB_sensor_channels[dataset[dataset_idx]],
                     'sliding_window_length': sliding_window_length[dataset[dataset_idx]],
                     'sliding_window_step': sliding_window_step[dataset[dataset_idx]],
                     #'num_attributes': num_attributes[dataset[dataset_idx]],
                     'batch_size_train': batch_size_train[network[network_idx]][dataset[dataset_idx]],
                     'batch_size_val': batch_size_val[network[network_idx]][dataset[dataset_idx]],
                     'num_tr_inputs': num_tr_inputs[dataset[dataset_idx]],
                     'num_classes': num_classes[dataset[dataset_idx]],
                     'label_pos': label_pos[2],
                     'file_suffix': 'results_yy{}mm{}dd{:02d}hh{:02d}mm{:02d}.xml'.format(now.year,
                                                                                          now.month,
                                                                                          now.day,
                                                                                          now.hour,
                                                                                          now.minute),
                     'dataset_root': dataset_root[dataset[dataset_idx]],
                     'accumulation_steps': accumulation_steps[dataset[dataset_idx]],
                     'reshape_input': reshape_input,
                     'name_counter': name_counter,
                     'freeze_options': freeze_options[freeze],
                     'percentages_names': percentages_names[percentage_idx],
                     'fully_convolutional': fully_convolutional,
                     'labeltype': labeltype,
                     'sacred':sacred,
                     'augmentations':augmentations[0],
                     'hidden_layer':lstm_hidlyr,
                     'layer_dim':lstm_lyrdim,
                     'trans_embed_layer': n_embedding_layers,
                     'transformer_dim': transformer_dim,
                     'transformer_heads': n_head,
                     'transformer_fc': dim_fc,
                     'transformer_layers': n_layers,
                     'trans_pos_embed': use_pos_embedding,
                     }

    return configuration


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
    config = configuration(dataset_idx=2,
                           network_idx=1,
                           output_idx=0,
                           usage_modus_idx=0,
                           #dataset_fine_tuning_idx=0,
                           reshape_input=False,
                           learning_rates_idx=0,
                           name_counter=0,
                           freeze=0,
                           fully_convolutional=False,
                           #percentage_idx=12,
                           #pooling=0
                           )
    
    dataset = config["dataset"]
    network = config["network"]
    output = config["output"]
    reshape_input = config["reshape_input"]
    usageModus = config["usage_modus"]
    #dataset_finetuning = config["dataset_finetuning"]
    #pooling = config["pooling"]
    lr = config["lr"]
    bsize = config["batch_size_train"]
    augmentation=config["augmentations"]
    
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
   
@ex.capture
def run(config, dataset, network, output, usageModus):
   
    file_name='/data/nnair/icpr2024/'
   
    file_name='/data/nnair/icpr2024/'+'logger.txt'
    
    setup_experiment_logger(logging_level=logging.DEBUG,filename=file_name)

    logging.info('Finished')
    logging.info('Dataset {} Network {} Output {} Modus {}'.format(dataset, network, output, usageModus))

    modus = Modus_Selecter(config, ex)

    # Starting process
    modus.net_modus()
    
    print("Done")


@ex.automain
def main():
    print("main began")
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

    print("Python  {}".format(platform.python_version()))


    run()

    print("Done")

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

