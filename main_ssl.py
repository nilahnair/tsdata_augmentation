#!/usr/bin/env python
# Based on https://github.com/facebookresearch/simsiam


import argparse
import builtins
import math
import os
from pathlib import Path
import sys
import random
import shutil
import time
import warnings
import datetime
import petname

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.models as models
from augmentations import get_augmentation
from har_dataset import HARDataset

from network import Network

from torch.utils.tensorboard import SummaryWriter
import yaml

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', metavar='DIR',
                    help='Name of dataset')
parser.add_argument('--augmentations', default=None, type=str,
                    help='Name of augmentations')


parser.add_argument('-a', '--arch', metavar='ARCH', default='cnn',
                    # choices=model_names,
                    help='model architecture: ' +
                        # ' | '.join(model_names) +
                        ' (default: cnn)')
parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=256, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--optimizer', default='sgd', choices=['sgd','adam'])
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--num_checkpoints', default=0, type=int,
                    help='Number of checkpoints to save during training. Saves every epochs // num_checkpoints.')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--output_dir', default='/data/amatei/experiments/icpr2024/har_augments/simsiam', type=str)

# simsiam specific configs:
parser.add_argument('--dim', default=256, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=128, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')
parser.add_argument('--polars_max_threads', default=4)

def main():
    args = parser.parse_args()

    os.environ['POLARS_MAX_THREADS'] = str(args.polars_max_threads)

    # args.output_dir = os.path.join(args.output_dir, datetime.datetime.now().strftime(f'%Y%m%d-%H-%M-%S-{petname.Generate()}'))
    # os.makedirs(args.output_dir, exist_ok=True)


    # sys.stdout = open(os.path.join(args.output_dir, 'run.log'), 'w') 

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))

    # logic to load backbone
    encoder_config = {
        'dataset': args.dataset,
        'reshape_input': False, #fixed
        'sliding_window_length': 100 if args.dataset == 'mbientlab' and args.arch != 'cnn_transformer' else 200,
        'fully_convolutional': 'FC', #fixed
        'filter_size': 5, #fixed
        'num_filters': 64, #fixed
        'transformer_dim': 64, #fixed
        'transformer_heads': 8, #fixed
        'transformer_fc': 128, #fixed
        'transformer_layers': 6, #fixed
        'trans_embed_layer': 4, #fixed
        'trans_pos_embed': True, #fixed
        'output': 'softmax', #fixed
        'num_classes': args.dim 
    }
    encoder_config['network'] = args.arch
    # backbone_config['NB_sensor_channels'] = {'mbientlab': 30, 'mocap': 126}.get(args.dataset, 9)
    encoder_config['NB_sensor_channels'] =  30 if args.dataset == 'mbientlab' else \
                                            126 if args.dataset == 'mocap'     else \
                                              9
    
    encoder = Network(encoder_config)

    model = SimSiam(base_encoder=encoder, dim=args.dim, pred_dim=args.pred_dim)

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    print(model) # print model after SyncBatchNorm

    # define loss function (criterion) and optimizer
    criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)

    if args.fix_pred_lr:
        optim_params = [{'params': model.encoder.parameters(), 'fix_lr': False},
                        {'params': model.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    match args.optimizer:
        case 'sgd':
            optimizer = torch.optim.SGD(optim_params, init_lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        case 'adam':
            optimizer = torch.optim.Adam(optim_params, init_lr, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.output_dir = str(Path(args.resume).parent)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        exp_identifier = '_'.join([args.dataset, args.arch, args.augmentations])
        args.output_dir = Path(args.output_dir) / exp_identifier / os.uname()[1] # add hostname to prevent merges from different hosts when copying
        run_id = 0
        while (args.output_dir / f'{run_id:03d}').exists():
            run_id +=1
        args.output_dir = args.output_dir / f'{run_id:03d}'
        args.output_dir.mkdir(parents=True)
        args.output_dir = str(args.output_dir)

    cudnn.benchmark = True

    # DONE add loadable augmentations here
    augmentation = get_augmentation(args.augmentations)

    # DONE add own dataloader here
    dataset_root_defaults = {
        'mocap':        "/vol/actrec/DFG_Project/2019/LARa_dataset/MoCap/LARa_dataset_mocap/",
        'mbientlab':    "/vol/actrec/DFG_Project/2019/LARa_dataset/Mbientlab/LARa_dataset_mbientlab/",
        'mobiact':      "/vol/actrec/MobiAct_Dataset/",
        'motionsense':  "/vol/actrec/motion-sense-master/data/A_DeviceMotion_data/A_DeviceMotion_data",
        'sisfall':      "/vol/actrec/SisFall_dataset"
        }
    # sliding_window_length_defaults = {}
    sliding_window_step_defaults = {'mocap': 25, 'mbientlab': 12, 'mobiact': 50, 'motionsense': 25, 'sisfall': 50}

    train_dataset = HARDataset(
        path=dataset_root_defaults[args.dataset],
        dataset_name=args.dataset,
        window_length=encoder_config['sliding_window_length'],
        window_stride=sliding_window_step_defaults[args.dataset],
        transform=TwoCropsTransform(augmentation),
        augmenation_probability=1.0,
        split='train'
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
        shuffle=True,
        drop_last=True)

    writer = SummaryWriter(log_dir=args.output_dir)

    if args.resume:
        log_step = args.start_epoch * len(train_loader)
    else:
        log_step = 0

    writer.add_hparams(hparam_dict=vars(args), metric_dict={})
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as cfile:
        yaml.dump(vars(args), cfile, default_flow_style=False)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        # train(train_loader, model, criterion, optimizer, epoch, args)
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses],
            prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        model.train()

        end = time.time()
        for i, batch in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            images = batch['data']
            if args.gpu is not None:
                images[0] = images[0].to(torch.float).cuda(args.gpu, non_blocking=True)
                images[1] = images[1].to(torch.float).cuda(args.gpu, non_blocking=True)

            # compute output and loss
            p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

            writer.add_scalar('loss', loss.item(), global_step=log_step)

            losses.update(loss.item(), images[0].size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
            
            log_step += 1

        save_path = f'checkpoint.pth.tar' 
        if args.output_dir != None:
            save_path = os.path.join(args.output_dir, save_path)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'dataset': args.dataset,
            'augmentations': args.augmentations,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'config': vars(args)
        }, is_best=False,
            filename=save_path)
        
        if args.num_checkpoints > 0 and epoch % (args.epochs // args.num_checkpoints) == 0:
            # save_path = f'checkpoint_{epoch:04d}.pth.tar' 
            # if args.output_dir != None:
            #     save_path = os.path.join(args.output_dir, save_path)
            # save_checkpoint({
            #     'epoch': epoch + 1,
            #     'arch': args.arch,
            #     'dataset': args.dataset,
            #     'augmentations': args.augmentations,
            #     'state_dict': model.state_dict(),
            #     'encoder_state_dict': model.encoder.state_dict(),
            #     'optimizer' : optimizer.state_dict(),
            #     'config': vars(args)
            # }, is_best=False,
            #     filename=save_path)

            save_path = f'encoder_{epoch:04d}.pth.tar' 
            if args.output_dir != None:
                save_path = os.path.join(args.output_dir, save_path)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'dataset': args.dataset,
                'augmentations': args.augmentations,
                'state_dict': model.encoder.state_dict(),
                'config': vars(args)
            }, is_best=False,
                filename=save_path)

    save_path = f'encoder.pth.tar' 
    if args.output_dir != None:
        save_path = os.path.join(args.output_dir, save_path)
    save_checkpoint({
        'arch': args.arch,
        'dataset': args.dataset,
        'augmentations': args.augmentations,
        'state_dict': model.encoder.state_dict(),
        'config': vars(args)
    }, is_best=False,
        filename=save_path)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512): # TODO sensible defaults for dims
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # MODIFIED instantiation of model outside of this class -> don't call constructor here
        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        # self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        self.encoder = base_encoder


        # modify fc3-5/imu_head here based on config['network']. config dict is accessable as part of the network.config
        # POSTPONED maybe add AdaptiveAvgPooling befor fc layer to decrease high prev_dim build a 3-layer projector
        #   -> no pooling necessary as flatten is already is applied 
        #   -> if problems arise, apply pooling before flatten
        match self.encoder.config['network']:
            case 'cnn': 
                # prev_dim = self.encoder.fc3.in_features
                # self.encoder.fc3 = torch.nn.Identity()
                self.encoder.fc3 = torch.nn.Linear(self.encoder.fc3.in_features, 2048)
                prev_dim = self.encoder.fc3.out_features
                self.encoder.fc4 = torch.nn.Identity()
            case 'cnn_imu':
                # prev_dim = self.encoder.fc4.in_features # toke fc4 because fc3 is per limb and is concatenated afterwards
                
                # self.encoder.fc3_LA = torch.nn.Identity()
                # self.encoder.fc3_LL = torch.nn.Identity()
                # self.encoder.fc3_RA = torch.nn.Identity()
                # self.encoder.fc3_RL = torch.nn.Identity()
                # self.encoder.fc3_N  = torch.nn.Identity()
                self.encoder.fc4    = torch.nn.Linear(self.encoder.fc4.in_features, 2048)
                prev_dim = self.encoder.fc4.out_features
                
                #concat after fc3 -> prev_dim ~= fc4.in_features # if to large, reduce in subsequnt layers
            case 'cnn_transformer':
                prev_dim = self.encoder.transformer_dim
                self.encoder.imu_head = torch.nn.Identity()

        # self.encoder.fc5 = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
        #                                 nn.BatchNorm1d(prev_dim),
        #                                 nn.ReLU(inplace=True), # first layer
        #                                 nn.Linear(prev_dim, prev_dim, bias=False),
        #                                 nn.BatchNorm1d(prev_dim),
        #                                 nn.ReLU(inplace=True), # second layer
        #                                 self.encoder.fc,
        #                                 nn.BatchNorm1d(dim, affine=False)) # output layer
        projection_layer = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False), # TODO if to many parameters because of high prev_dim, reduce to dim here
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        # self.encoder.fc,
                                        nn.Linear(prev_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        if self.encoder.config['network'] in ['cnn', 'cnn_imu']:
            self.encoder.fc5 = projection_layer
        else: # else cnn_transformer
            self.encoder.imu_head = projection_layer
        #self.encoder.fc5[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # TODO ideally nothing to change here
        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x): # TODO handle what happens when transform is a list
        q = self.base_transform(x)
        
        # if augmentation is deterministic, only apply on query
        if self.base_transform.__name__ not in ['flipping', 'vertical_flip']:
            k = self.base_transform(x)
        else:
            k = x

        return [q, k]


if __name__ == '__main__':
    main()
