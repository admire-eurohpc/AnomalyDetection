import time
import logging
import torch
import torch.nn as nn
import numpy as np
from sequitur.models.lstm_ae import LSTM_AE
from sequitur.models.conv_ae import CONV_AE

class Encoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        """
        Args:
           num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        kernel = (1, 3)
        padding = (0, 1)
        stride = (1, 2)
        
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=kernel, padding=padding, stride=stride),  # 16x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, 2* c_hid, kernel_size=kernel, padding=padding, stride=stride),  # 16x16 => 16x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 4 * c_hid, kernel_size=kernel, padding=padding, stride=stride),  # 16x8 => 16x4
            act_fn(),
            nn.Conv2d(4 * c_hid, 8 * c_hid, kernel_size=kernel, padding=padding, stride=stride), 
            act_fn(),
            nn.Flatten(),
            #nn.Linear(10 * 5 * c_hid * 4, latent_dim),
        )
        

    def forward(self, x):
        logging.debug(f'Encoder inference shape: {x.shape}')
        return self.net(x)


class CNN_encoder(nn.Module):
    def __init__(self, kernel_size: int, latent_dim: int = 4, cpu_alloc: bool = False):
        """
        Args:
           num_input_channels : Number of input channels 3 without cpus_alloc, 4 with this feature
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        #assuming window size of 60
        if cpu_alloc:
            kernel_size2d = (4, kernel_size)
            maxpool_size2d = (4,3)
        else:
            kernel_size2d = (3, kernel_size)
            maxpool_size2d = 3
        
        modules = []
        channels = [1, 8, 32]
        modules.append(nn.Conv2d(channels[0], channels[0], kernel_size=kernel_size2d, padding='same')) #input = (200x3x60), output = (200, 3, 60)
        modules.append(nn.ReLU())
        modules.append(nn.Conv2d(channels[0], channels[1], kernel_size=kernel_size2d, padding='same')) #input = (200x3x60), output = (400, 3, 60)
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool2d(kernel_size=(maxpool_size2d))) #input = (400x3x60), output = (400, 1, 20)
        modules.append(nn.Flatten(start_dim=2))
        modules.append(nn.Conv1d(channels[1], channels[2], kernel_size=kernel_size, padding='same'))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool1d(4))
        modules.append(nn.Flatten())
        modules.append(nn.Linear(latent_dim*40, latent_dim)) # latent dim 4

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        #logging.debug(f'Encoder inference shape: {x.shape}')
        return self.model(x)

class CNN_LSTM_encoder(nn.Module):
    def __init__(self, kernel_size: int=10, 
                 lstm_input_dim: int=40, 
                 lstm_out_dim: int=10,
                 h_lstm_chan: list=[32, 64], 
                 cpu_alloc: bool=True,) -> None:
        super().__init__()

        if cpu_alloc: input_channels = 4
        else: input_channels = 3

        
        #CNN encoder
        cnn_modules = []
        channels = [8, 16]
        cnn_modules.append(nn.Conv1d(input_channels, channels[0], kernel_size=kernel_size, padding='same')) #input = (batch x channel[0] x 60), output = (batch x channel[1] x 60)
        cnn_modules.append(nn.ReLU())
        cnn_modules.append(nn.AvgPool1d(3)) # input = (N x channel[0] x 60), output = (N x channel[0] x 20)
        cnn_modules.append(nn.Dropout(0.2))

        cnn_modules.append(nn.Conv1d(channels[0], channels[1], kernel_size=kernel_size, padding='same')) #input = (batch x channel[1] x 20), output = (batch x channel[2] x 20)
        cnn_modules.append(nn.ReLU())
        cnn_modules.append(nn.AvgPool1d(4)) # input = (N x channel[2] x 20), output = (N x channel[2] x 5)
        cnn_modules.append(nn.Dropout(0.2))

        cnn_modules.append(nn.Flatten())
        cnn_modules.append(nn.Linear(80, 40)) # cnn_encoding_len 40

        self.cnn_encoder = nn.Sequential(*cnn_modules)

        #LSTM encoder 1st approach
        layer_dims = [lstm_input_dim] + h_lstm_chan + [lstm_out_dim]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=1,
                batch_first=True
            )
            self.layers.append(layer)
        self.h_activ, self.out_activ = nn.Sigmoid(), nn.Tanh()
        
    def forward(self, x):
        x = self.cnn_encoder(x)
        x = x.unsqueeze(2)
        for index, layer in enumerate(self.layers):
            x, (h_n, c_n) = layer(x)
            if self.h_activ and index < self.num_layers - 1:
                x = self.h_activ(x)
            elif self.out_activ and index == self.num_layers - 1:
                return self.out_activ(h_n).squeeze(0)
        return h_n.squeeze()