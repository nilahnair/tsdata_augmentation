'''
Created on May 17, 2019

@author: fmoya
'''

from __future__ import print_function
import logging


import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np


class Network(nn.Module):
    '''
    classdocs
    '''


    def __init__(self, config):
        '''
        Constructor
        '''
        
        super(Network, self).__init__()
        
        logging.info('            Network: Constructor')
        
        self.config = config

        if self.config["reshape_input"]:
            in_channels = 3
            Hx = int(self.config['NB_sensor_channels'] / 3)
        else:
            in_channels = 1
            Hx = self.config['NB_sensor_channels']
        Wx = self.config['sliding_window_length']

        if self.config["fully_convolutional"] == "FCN":
            padding = [2, 0]
        elif self.config["fully_convolutional"] == "FC":
            padding = 0


        # Computing the size of the feature maps
        Wx, Hx = self.size_feature_map(Wx=Wx,
                                       Hx=Hx,
                                       F=(self.config['filter_size'], 1),
                                       P=padding, S=(1, 1), type_layer='conv')
        logging.info('            Network: Wx {} and Hx {}'.format(Wx, Hx))
        Wx, Hx = self.size_feature_map(Wx=Wx,
                                       Hx=Hx,
                                       F=(self.config['filter_size'], 1),
                                       P=padding, S=(1, 1), type_layer='conv')
        logging.info('            Network: Wx {} and Hx {}'.format(Wx, Hx))
        Wx, Hx = self.size_feature_map(Wx=Wx,
                                       Hx=Hx,
                                       F=(self.config['filter_size'], 1),
                                       P=padding, S=(1, 1), type_layer='conv')
        logging.info('            Network: Wx {} and Hx {}'.format(Wx, Hx))
        Wx, Hx = self.size_feature_map(Wx=Wx,
                                       Hx=Hx,
                                       F=(self.config['filter_size'], 1),
                                       P=padding, S=(1, 1), type_layer='conv')
        logging.info('            Network: Wx {} and Hx {}'.format(Wx, Hx))
        
        if self.config["network"] == "lstm":
            self.hidden_dim = self.config['hidden_layer']
            self.layer_dim = self.config['layer_dim']
            self.rnn = nn.LSTM(self.config['NB_sensor_channels'], self.hidden_dim, self.layer_dim, batch_first=True)
            self.fc3 = nn.Linear(self.config['hidden_layer'] * self.config['sliding_window_length'], 256)
            self.batch_size = None
            self.hidden = None
        elif self.config["network"]=="cnn_transformer":
            self.input_dim = self.config['NB_sensor_channels']
            self.output_dim = self.config['num_classes']
            self.window_size = self.config['sliding_window_length']
            self.n_embedding_layer= self.config['trans_embed_layer']
            transformer_dim=self.config["transformer_dim"]
            self.transformer_dim = transformer_dim if self.n_embedding_layer > 0 else self.input_dim
            self.n_head = get_nhead(self.transformer_dim, self.config['transformer_heads'])
            self.dim_fc = self.config['transformer_fc']
            self.n_layers = self.config['transformer_layers']
            self.use_pos_embedding = self.config['trans_pos_embed']
            activation_function='gelu'
            self.activation_function = nn.GELU() if activation_function.lower() == 'gelu' else nn.ReLU()
            
            
            
        # set the Conv layers
        if self.config["network"] == "cnn":
            self.conv1_1 = nn.Conv2d(in_channels=in_channels,
                                     out_channels=self.config['num_filters'],
                                     kernel_size=(self.config['filter_size'], 1),
                                     stride=1, padding=padding)
            self.conv1_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                     out_channels=self.config['num_filters'],
                                     kernel_size=(self.config['filter_size'], 1),
                                     stride=1, padding=padding)
            self.conv2_1 = nn.Conv2d(in_channels=self.config['num_filters'],
                                     out_channels=self.config['num_filters'],
                                     kernel_size=(self.config['filter_size'], 1),
                                     stride=1, padding=padding)
            self.conv2_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                     out_channels=self.config['num_filters'],
                                     kernel_size=(self.config['filter_size'], 1),
                                     stride=1, padding=padding)

            if self.config["fully_convolutional"] == "FCN":
                if self.config["reshape_input"]:
                    self.fc3 = nn.Conv2d(in_channels=self.config['num_filters'],
                                         out_channels=256, kernel_size=(1, 1), stride=1, padding=0)
                else:
                    self.fc3 = nn.Conv2d(in_channels=self.config['num_filters'],
                                         out_channels=256, kernel_size=(1, 1), stride=1, padding=0)
            elif self.config["fully_convolutional"] == "FC":
                if self.config["reshape_input"]:
                    self.fc3 = nn.Linear(self.config['num_filters'] *
                                         int(Wx) * int(self.config['NB_sensor_channels'] / 3), 256)
                else:
                    self.fc3 = nn.Linear(self.config['num_filters'] * int(Wx) * self.config['NB_sensor_channels'], 256)

        # set the Conv layers
        if self.config["network"] == "cnn_imu":
            # LA
            self.conv_LA_1_1 = nn.Conv2d(in_channels=in_channels,
                                     out_channels=self.config['num_filters'],
                                     kernel_size=(self.config['filter_size'], 1),
                                     stride=1, padding=padding)

            self.conv_LA_1_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padding)

            self.conv_LA_2_1 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padding)

            self.conv_LA_2_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padding)

            if self.config["reshape_input"]:
                if self.config["NB_sensor_channels"] == 27:
                    self.fc3_LA = nn.Linear(self.config['num_filters'] * int(Wx) *
                                            int(self.config['NB_sensor_channels'] / 9), 256)
                elif self.config["NB_sensor_channels"] == 30:
                    self.fc3_LA = nn.Linear(self.config['num_filters'] * int(Wx) *
                                            int(self.config['NB_sensor_channels'] / 15), 256)
                elif self.config["NB_sensor_channels"] == 126:
                    self.fc3_LA = nn.Linear(self.config['num_filters'] * int(Wx) * 10, 256)
            else:
                if self.config["NB_sensor_channels"] == 27:
                    self.fc3_LA = nn.Linear(self.config['num_filters'] * int(Wx) *
                                            int(self.config['NB_sensor_channels'] / 3), 256)
                elif self.config["NB_sensor_channels"] == 30:
                    self.fc3_LA = nn.Linear(self.config['num_filters'] * int(Wx) *
                                            int(self.config['NB_sensor_channels'] / 5), 256)
                elif self.config["NB_sensor_channels"] == 126:
                    self.fc3_LA = nn.Linear(self.config['num_filters'] * int(Wx) * 30, 256)

            # LL
            self.conv_LL_1_1 = nn.Conv2d(in_channels=in_channels,
                                     out_channels=self.config['num_filters'],
                                     kernel_size=(self.config['filter_size'], 1),
                                     stride=1, padding=padding)

            self.conv_LL_1_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padding)

            self.conv_LL_2_1 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padding)

            self.conv_LL_2_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padding)

            if self.config["reshape_input"]:
                if self.config["NB_sensor_channels"] == 30:
                    self.fc3_LL = nn.Linear(self.config['num_filters'] * int(Wx) *
                                            int(self.config['NB_sensor_channels'] / 15), 256)
                elif self.config["NB_sensor_channels"] == 126:
                    self.fc3_LL = nn.Linear(self.config['num_filters'] * int(Wx) * 8, 256)
            else:
                if self.config["NB_sensor_channels"] == 30:
                    self.fc3_LL = nn.Linear(self.config['num_filters'] * int(Wx) *
                                            int(self.config['NB_sensor_channels'] / 5), 256)
                elif self.config["NB_sensor_channels"] == 126:
                    self.fc3_LL = nn.Linear(self.config['num_filters'] * int(Wx) * 24, 256)

            # N
            self.conv_N_1_1 = nn.Conv2d(in_channels=in_channels,
                                     out_channels=self.config['num_filters'],
                                     kernel_size=(self.config['filter_size'], 1),
                                     stride=1, padding=padding)

            self.conv_N_1_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padding)

            self.conv_N_2_1 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padding)

            self.conv_N_2_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padding)

            if self.config["reshape_input"]:
                if self.config["NB_sensor_channels"] == 27:
                    self.fc3_N = nn.Linear(self.config['num_filters'] * int(Wx) *
                                           int(self.config['NB_sensor_channels'] / 9), 256)
                elif self.config["NB_sensor_channels"] == 30:
                    self.fc3_N = nn.Linear(self.config['num_filters'] * int(Wx) *
                                           int(self.config['NB_sensor_channels'] / 15), 256)
                elif self.config["NB_sensor_channels"] == 126:
                    self.fc3_N = nn.Linear(self.config['num_filters'] * int(Wx) * 6, 256)
            else:
                if self.config["NB_sensor_channels"] == 27:
                    self.fc3_N = nn.Linear(self.config['num_filters'] * int(Wx) *
                                           int(self.config['NB_sensor_channels'] / 3), 256)
                elif self.config["NB_sensor_channels"] == 30:
                    self.fc3_N = nn.Linear(self.config['num_filters'] * int(Wx) *
                                           int(self.config['NB_sensor_channels'] / 5), 256)
                elif self.config["NB_sensor_channels"] == 126:
                    self.fc3_N = nn.Linear(self.config['num_filters'] * int(Wx) * 18, 256)


            # RA
            self.conv_RA_1_1 = nn.Conv2d(in_channels=in_channels,
                                     out_channels=self.config['num_filters'],
                                     kernel_size=(self.config['filter_size'], 1),
                                     stride=1, padding=padding)

            self.conv_RA_1_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padding)

            self.conv_RA_2_1 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padding)

            self.conv_RA_2_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padding)

            if self.config["reshape_input"]:
                if self.config["NB_sensor_channels"] == 27:
                    self.fc3_RA = nn.Linear(self.config['num_filters'] * int(Wx) *
                                            int(self.config['NB_sensor_channels'] / 9), 256)
                elif self.config["NB_sensor_channels"] == 30:
                    self.fc3_RA = nn.Linear(self.config['num_filters'] * int(Wx) *
                                            int(self.config['NB_sensor_channels'] / 15), 256)
                elif self.config["NB_sensor_channels"] == 126:
                    self.fc3_RA = nn.Linear(self.config['num_filters'] * int(Wx) * 10, 256)
            else:
                if self.config["NB_sensor_channels"] == 27:
                    self.fc3_RA = nn.Linear(self.config['num_filters'] * int(Wx) *
                                            int(self.config['NB_sensor_channels'] / 3), 256)
                elif self.config["NB_sensor_channels"] == 30:
                    self.fc3_RA = nn.Linear(self.config['num_filters'] * int(Wx) *
                                            int(self.config['NB_sensor_channels'] / 5), 256)
                elif self.config["NB_sensor_channels"] == 126:
                    self.fc3_RA = nn.Linear(self.config['num_filters'] * int(Wx) * 30, 256)

            # RL
            self.conv_RL_1_1 = nn.Conv2d(in_channels=in_channels,
                                     out_channels=self.config['num_filters'],
                                     kernel_size=(self.config['filter_size'], 1),
                                     stride=1, padding=padding)

            self.conv_RL_1_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padding)

            self.conv_RL_2_1 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padding)

            self.conv_RL_2_2 = nn.Conv2d(in_channels=self.config['num_filters'],
                                          out_channels=self.config['num_filters'],
                                          kernel_size=(self.config['filter_size'], 1),
                                          stride=1, padding=padding)

            if self.config["reshape_input"]:
                if self.config["NB_sensor_channels"] == 30:
                    self.fc3_RL = nn.Linear(self.config['num_filters'] * int(Wx) *
                                            int(self.config['NB_sensor_channels'] / 15), 256)
                elif self.config["NB_sensor_channels"] == 126:
                    self.fc3_RL = nn.Linear(self.config['num_filters'] * int(Wx) * 8, 256)
            else:
                if self.config["NB_sensor_channels"] == 30:
                    self.fc3_RL = nn.Linear(self.config['num_filters'] * int(Wx) *
                                            int(self.config['NB_sensor_channels'] / 5), 256)
                elif self.config["NB_sensor_channels"] == 126:
                    self.fc3_RL = nn.Linear(self.config['num_filters'] * int(Wx) * 24, 256)
        
        if self.config["network"]=="cnn_transformer":
            self.input_proj = nn.ModuleList()
            for _ in range(self.n_embedding_layer):
                d_in = self.input_dim if len(self.input_proj) == 0 else self.transformer_dim
                conv_layer = nn.Sequential(nn.Conv1d(d_in, self.transformer_dim, 1), self.activation_function)
                self.input_proj.append(conv_layer)
            
            #setting parameters
            self.cls_token = nn.Parameter(torch.zeros((1, self.transformer_dim)))
        
            #setting positional encoding
            if self.use_pos_embedding:
                self.position_embed = nn.Parameter(torch.randn(self.window_size + 1, 1, self.transformer_dim))
        
            #set transformer encoder
            encoder_layer = TransformerEncoderLayer(d_model = self.transformer_dim, nhead = self.n_head, dim_feedforward = self.dim_fc,
                                       dropout = 0.1, activation = 'gelu')
            self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers = self.n_layers, norm = nn.LayerNorm(self.transformer_dim))
        
            #setting mlp layers
            self.imu_head = nn.Sequential(nn.LayerNorm(self.transformer_dim), nn.Linear(self.transformer_dim, self.transformer_dim//4),
                                      self.activation_function, nn.Dropout(0.1), nn.Linear(self.transformer_dim//4, self.output_dim))
        else:# MLP
            if self.config["fully_convolutional"] == "FCN":
                if self.config["network"] == "cnn":
                    self.fc4 = nn.Conv2d(in_channels=256,
                                     out_channels=256, kernel_size=(1, 1), stride=1, padding=0)
                elif self.config["network"] == "cnn_imu" and self.config["NB_sensor_channels"] in [30, 126]:
                    self.fc4 = nn.Conv2d(in_channels=256 * 5,
                                     out_channels=256, kernel_size=(1, 1), stride=1, padding=0)
                elif self.config["network"] == "cnn_imu" and self.config["NB_sensor_channels"] == 27:
                    self.fc4 = nn.Conv2d(in_channels=256 * 3,
                                     out_channels=256, kernel_size=(1, 1), stride=1, padding=0)
            elif self.config["fully_convolutional"] == "FC":
                if self.config["network"] == "cnn":
                    self.fc4 = nn.Linear(256, 256)
                elif self.config["network"] == "cnn_imu" and self.config["NB_sensor_channels"] in [30, 126]:
                    self.fc4 = nn.Linear(256 * 5, 256)
                elif self.config["network"] == "cnn_imu" and self.config["NB_sensor_channels"] == 27:
                    self.fc4 = nn.Linear(256 * 3, 256)
                elif self.config["network"]=="lstm":
                    self.fc4 = nn.Linear(256, 256)

            if self.config["fully_convolutional"] == "FCN":
                if self.config['output'] == 'softmax':
                    self.fc5 = nn.Conv2d(in_channels=256,
                                     out_channels=self.config['num_classes'], kernel_size=(1, 1), stride=1, padding=0)
                elif self.config['output'] == 'attribute':
                    self.fc5 = nn.Conv2d(in_channels=256,
                                     out_channels=self.config['num_attributes'],
                                     kernel_size=(1, 1), stride=1, padding=0)
                elif self.config['output'] == 'identity':
                    self.fc5 = nn.Conv2d(in_channels=256,
                                     out_channels=self.config['num_classes'], kernel_size=(1, 1), stride=1, padding=0)
            elif self.config["fully_convolutional"] == "FC":
                if self.config['output'] == 'softmax':
                    self.fc5 = nn.Linear(256, self.config['num_classes'])
                elif self.config['output'] == 'attribute':
                    self.fc5 = nn.Linear(256, self.config['num_attributes'])
                elif self.config['output'] == 'identity':
                    self.fc5 = nn.Linear(256, self.config['num_classes'])

        self.avgpool = nn.AvgPool2d(kernel_size=[1, self.config['NB_sensor_channels']])

        self.softmax = nn.Softmax()

        self.sigmoid = nn.Sigmoid()

        return


    def forward(self, x):
        '''
        Forwards function, required by torch.

        @param x: batch [batch, 1, Channels, Time], Channels = Sensors * 3 Axis
        @return x: Output of the network, either Softmax or Attribute
        '''

        # Reshaping the input sequence to [batch, 3, Sensors, Time]
        if self.config["reshape_input"]:
            x = x.permute(0, 2, 1, 3)
            x = x.view(x.size()[0], x.size()[1], int(x.size()[3] / 3), 3)
            x = x.permute(0, 3, 1, 2)

        # Selecting the one ot the two networks, tCNN or tCNN-IMU
        if self.config["network"] == "cnn":
            x = self.tcnn(x)
        elif self.config["network"] == "cnn_imu":
            if self.config["dataset"] in ['motionminers_real', 'motionminers_flw']:
                x_LA, x_N, x_RA = self.tcnn_imu(x)
                x = torch.cat((x_LA, x_N, x_RA), 1)
            else:
                x_LA, x_LL, x_N, x_RA, x_RL = self.tcnn_imu(x)
                x = torch.cat((x_LA, x_LL, x_N, x_RA, x_RL), 1)
        elif self.config["network"]=="lstm":
            x = x.permute(0,2,3,1)
            x = x.view(x.size()[0], x.size()[1], x.size()[2])
            x=self.lstm(x)
            x = x.reshape((-1, x.size()[1] * x.size()[2]))
            x=self.fc3(x)
            
            
        if self.config["network"]=="cnn_transformer":
            #here[B,1,Win,D] to [B,Win,D,1]
            x=x.permute(0,2,3,1)
            #to [B,D,Win]
            x = x.view(x.size()[0], x.size()[1], x.size()[2])
            #testing if the below line makes things work
            x=x.permute(0,2,1)
            #input embedding
            for conv_layer in self.input_proj:
                x = conv_layer(x)
            
            # Reshaping: [B, D', Win] -> [Win, B, D'] 
            x = x.permute(2, 0, 1)
        
            # Prepend class token: [Win, B, D']  -> [Win+1, B, D']
            cls_token = self.cls_token.unsqueeze(1).repeat(1, x.shape[1], 1)
            x = torch.cat([cls_token, x])
        
            #position embedding
            if self.use_pos_embedding:
                x += self.position_embed
            
            #transformer
            # Transformer Encoder pass
            x = self.transformer_encoder(x)[0]
            # Pass through fully-connected layers
            x= self.imu_head(x)
            
            # Selecting MLP, either FC or FCN
        elif self.config["fully_convolutional"] == "FCN":
            x = F.dropout(x, training=self.training)
            x = F.relu(self.fc4(x))
            x = F.dropout(x, training=self.training)
            x = self.fc5(x)
            x = self.avgpool(x)
            x = x.view(x.size()[0], x.size()[1], x.size()[2])
            x = x.permute(0, 2, 1)
        elif self.config["fully_convolutional"] == "FC":
            x = F.dropout(x, training=self.training)
            x = F.relu(self.fc4(x))
            x = F.dropout(x, training=self.training)
            x = self.fc5(x)
            
        if self.config['output'] == 'attribute':
            x = self.sigmoid(x)

        if not self.training:
            if self.config['output'] == 'softmax' or self.config['output'] == 'identity':
                x = self.softmax(x)
               

        return x
        #return x11.clone(), x12.clone(), x21.clone(), x22.clone(), x


    def init_weights(self):
        '''
        Applying initialisation of layers
        '''
        if self.config['network']=="cnn_transformer":
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        else:
            self.apply(Network._init_weights_orthonormal)
        return
    
    def __str__(self):
        return 'Transformer_Encoder with dim={} & activation={}'.format(self.transformer_dim, str(self.activation_function))

    @staticmethod
    def _init_weights_orthonormal(m):
        '''
        Orthonormal Initialissation of layer

        @param m: layer m
        '''
        if isinstance(m, nn.Conv2d):
            #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #m.weight.data.normal_(0, (2. / n) ** (1 / 2.0))
            nn.init.orthogonal_(m.weight, gain = np.sqrt(2))
            nn.init.constant_(m.bias.data, 0)
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain = np.sqrt(2))
            nn.init.constant_(m.bias.data, 0)

        return


    def size_feature_map(self, Wx, Hx, F, P, S, type_layer = 'conv'):
        '''
        Computing size of feature map after convolution or pooling

        @param Wx: Width input
        @param Hx: Height input
        @param F: Filter size
        @param P: Padding
        @param S: Stride
        @param type_layer: conv or pool
        @return Wy: Width output
        @return Hy: Height output
        '''

        if self.config["fully_convolutional"] == "FCN":
            Pw = P[0]
            Ph = P[1]
        elif self.config["fully_convolutional"] == "FC":
            Pw = P
            Ph = P

        if type_layer == 'conv':
            Wy = 1 + (Wx - F[0] + 2 * Pw) / S[0]
            Hy = 1 + (Hx - F[1] + 2 * Ph) / S[1]

        elif type_layer == 'pool':
            Wy = 1 + (Wx - F[0]) / S[0]
            Hy = 1 + (Hx - F[1]) / S[1]

        return Wy, Hy
    
    def lstm(self, x):
        '''
        lstm network

        @param x: input sequence
        @return x: Prediction of the network
        '''
        h0, c0 = self.init_hidden(x)
        x, (hn, cn) = self.rnn(x, (h0, c0))
        #out = self.fc(out[:, -1, :])
        return x
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]


    def tcnn(self, x):
        '''
        tCNN network

        @param x: input sequence
        @return x: Prediction of the network
        '''
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        # x12 = F.max_pool2d(x12, (2, 1))

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        # x = F.max_pool2d(x, (2, 1))

        if self.config["fully_convolutional"] == "FCN":
            x = F.relu(self.fc3(x))
        elif self.config["fully_convolutional"] == "FC":
            # view is reshape
            x = x.reshape((-1, x.size()[1] * x.size()[2] * x.size()[3]))
            x = F.relu(self.fc3(x))
        return x


    def tcnn_imu(self, x):
        '''
        tCNN-IMU network
        The parameters will adapt according to the dataset, reshape and output type

        x_LA, x_LL, x_N, x_RA, x_RL

        @param x: input sequence
        @return x_LA: Features from left arm
        @return x_LL: Features from left leg
        @return x_N: Features from Neck or Torso
        @return x_RA: Features from Right Arm
        @return x_RL: Features from Right Leg
        '''
        # LA
        if self.config["reshape_input"]:
            if self.config["NB_sensor_channels"] == 27:
                x_LA = F.relu(self.conv_LA_1_1(x[:, :, :, 0:3]))
            if self.config["NB_sensor_channels"] == 30:
                x_LA = F.relu(self.conv_LA_1_1(x[:, :, :, 0:2]))
            elif self.config["NB_sensor_channels"] == 126:
                idx_LA = np.arange(4, 8)
                idx_LA = np.concatenate([idx_LA, np.arange(12, 14)])
                idx_LA = np.concatenate([idx_LA, np.arange(18, 22)])
                x_LA = F.relu(self.conv_LA_1_1(x[:, :, :, idx_LA]))
        else:
            if self.config["NB_sensor_channels"] == 27:
                x_LA = F.relu(self.conv_LA_1_1(x[:, :, :, 0:9]))
            if self.config["NB_sensor_channels"] == 30:
                x_LA = F.relu(self.conv_LA_1_1(x[:, :, :, 0:6]))
            elif self.config["NB_sensor_channels"] == 126:
                idx_LA = np.arange(12, 24)
                idx_LA = np.concatenate([idx_LA, np.arange(36, 42)])
                idx_LA = np.concatenate([idx_LA, np.arange(54, 66)])
                x_LA = F.relu(self.conv_LA_1_1(x[:, :, :, idx_LA]))

        x_LA = F.relu(self.conv_LA_1_2(x_LA))
        x_LA = F.relu(self.conv_LA_2_1(x_LA))
        x_LA = F.relu(self.conv_LA_2_2(x_LA))
        # view is reshape
        x_LA = x_LA.reshape(-1, x_LA.size()[1] * x_LA.size()[2] * x_LA.size()[3])
        x_LA = F.relu(self.fc3_LA(x_LA))

        # LL
        if self.config["NB_sensor_channels"] in [30, 126]:
            if self.config["reshape_input"]:
                if self.config["NB_sensor_channels"] == 30:
                    x_LL = F.relu(self.conv_LL_1_1(x[:, :, :, 2:4]))
                elif self.config["NB_sensor_channels"] == 126:
                    idx_LL = np.arange(8, 12)
                    idx_LL = np.concatenate([idx_LL, np.arange(14, 18)])
                    x_LL = F.relu(self.conv_LA_1_1(x[:, :, :, idx_LL]))
            else:
                if self.config["NB_sensor_channels"] == 30:
                    x_LL = F.relu(self.conv_LL_1_1(x[:, :, :, 6:12]))
                elif self.config["NB_sensor_channels"] == 126:
                    idx_LL = np.arange(24, 36)
                    idx_LL = np.concatenate([idx_LL, np.arange(42, 54)])
                    x_LL = F.relu(self.conv_LA_1_1(x[:, :, :, idx_LL]))

            x_LL = F.relu(self.conv_LL_1_2(x_LL))
            x_LL = F.relu(self.conv_LL_2_1(x_LL))
            x_LL = F.relu(self.conv_LL_2_2(x_LL))
            # view is reshape
            x_LL = x_LL.reshape(-1, x_LL.size()[1] * x_LL.size()[2] * x_LL.size()[3])
            x_LL = F.relu(self.fc3_LL(x_LL))

        # N
        if self.config["reshape_input"]:
            if self.config["NB_sensor_channels"] == 27:
                x_N = F.relu(self.conv_N_1_1(x[:, :, :, 3:6]))
            if self.config["NB_sensor_channels"] == 30:
                x_N = F.relu(self.conv_N_1_1(x[:, :, :, 4:6]))
            elif self.config["NB_sensor_channels"] == 126:
                idx_N = np.arange(0, 4)
                idx_N = np.concatenate([idx_N, np.arange(40, 42)])
                x_N = F.relu(self.conv_LA_1_1(x[:, :, :, idx_N]))
        else:
            if self.config["NB_sensor_channels"] == 27:
                x_N = F.relu(self.conv_N_1_1(x[:, :, :, 9:18]))
            if self.config["NB_sensor_channels"] == 30:
                x_N = F.relu(self.conv_N_1_1(x[:, :, :, 12:18]))
            elif self.config["NB_sensor_channels"] == 126:
                idx_N = np.arange(0, 12)
                idx_N = np.concatenate([idx_N, np.arange(120, 126)])
                x_N = F.relu(self.conv_LA_1_1(x[:, :, :, idx_N]))
        x_N = F.relu(self.conv_N_1_2(x_N))
        x_N = F.relu(self.conv_N_2_1(x_N))
        x_N = F.relu(self.conv_N_2_2(x_N))
        # view is reshape
        x_N = x_N.reshape(-1, x_N.size()[1] * x_N.size()[2] * x_N.size()[3])
        x_N = F.relu(self.fc3_N(x_N))

        # RA
        if self.config["reshape_input"]:
            if self.config["NB_sensor_channels"] == 27:
                x_RA = F.relu(self.conv_RA_1_1(x[:, :, :, 6:9]))
            if self.config["NB_sensor_channels"] == 30:
                x_RA = F.relu(self.conv_RA_1_1(x[:, :, :, 6:8]))
            elif self.config["NB_sensor_channels"] == 126:
                idx_RA = np.arange(22, 26)
                idx_RA = np.concatenate([idx_RA, np.arange(30, 32)])
                idx_RA = np.concatenate([idx_RA, np.arange(36, 40)])
                x_RA = F.relu(self.conv_LA_1_1(x[:, :, :, idx_RA]))
        else:
            if self.config["NB_sensor_channels"] == 27:
                x_RA = F.relu(self.conv_RA_1_1(x[:, :, :, 18:27]))
            if self.config["NB_sensor_channels"] == 30:
                x_RA = F.relu(self.conv_RA_1_1(x[:, :, :, 18:24]))
            elif self.config["NB_sensor_channels"] == 126:
                idx_RA = np.arange(66, 78)
                idx_RA = np.concatenate([idx_RA, np.arange(90, 96)])
                idx_RA = np.concatenate([idx_RA, np.arange(108, 120)])
                x_RA = F.relu(self.conv_LA_1_1(x[:, :, :, idx_RA]))

        x_RA = F.relu(self.conv_RA_1_2(x_RA))
        x_RA = F.relu(self.conv_RA_2_1(x_RA))
        x_RA = F.relu(self.conv_RA_2_2(x_RA))
        # view is reshape
        x_RA = x_RA.reshape(-1, x_RA.size()[1] * x_RA.size()[2] * x_RA.size()[3])
        x_RA = F.relu(self.fc3_RA(x_RA))

        # RL
        if self.config["NB_sensor_channels"] in [30, 126]:
            if self.config["reshape_input"]:
                if self.config["NB_sensor_channels"] == 30:
                    x_RL = F.relu(self.conv_RL_1_1(x[:, :, :, 8:10]))
                elif self.config["NB_sensor_channels"] == 126:
                    idx_RL = np.arange(26, 30)
                    idx_RL = np.concatenate([idx_RL, np.arange(32, 36)])
                    x_RL = F.relu(self.conv_LA_1_1(x[:, :, :, idx_RL]))
            else:
                if self.config["NB_sensor_channels"] == 30:
                    x_RL = F.relu(self.conv_RL_1_1(x[:, :, :, 24:30]))
                elif self.config["NB_sensor_channels"] == 126:
                    idx_RL = np.arange(78, 90)
                    idx_RL = np.concatenate([idx_RL, np.arange(96, 108)])
                    x_RL = F.relu(self.conv_LA_1_1(x[:, :, :, idx_RL]))

            x_RL = F.relu(self.conv_RL_1_2(x_RL))
            x_RL = F.relu(self.conv_RL_2_1(x_RL))
            x_RL = F.relu(self.conv_RL_2_2(x_RL))
            # view is reshape
            x_RL = x_RL.reshape(-1, x_RL.size()[1] * x_RL.size()[2] * x_RL.size()[3])
            x_RL = F.relu(self.fc3_RL(x_RL))

        if self.config["NB_sensor_channels"] == 27:
            return x_LA, x_N, x_RA
        else:
            return x_LA, x_LL, x_N, x_RA, x_RL
        
def get_nhead(embed_dim, n_head):
    for hd in range(n_head, 0, -1):
        if embed_dim % hd == 0:
            logging.info('N_head = {}'.format(hd))
            return hd
