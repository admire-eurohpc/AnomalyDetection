import logging
import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        """
        Args:
           num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        kernel = (1, 3)
        stride = (1, 2)
        
        self.linear = nn.Sequential(nn.Linear(latent_dim, 10 * 5 * c_hid), act_fn())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=kernel, output_padding=1, padding=1, stride=stride),  # 10x5 => 10x12
            act_fn(),
            nn.Conv2d(16, 16, kernel_size=kernel, padding=1),
            act_fn(),
            nn.ConvTranspose2d(16, 8, kernel_size=kernel, output_padding=1, padding=1, stride=stride),  
            act_fn(),
            nn.Conv2d(8, 8, kernel_size=kernel, padding=1),
            act_fn(),
            nn.ConvTranspose2d(8, num_input_channels, kernel_size=kernel, output_padding=1, padding=1, stride=stride),  
            nn.ReLU(),  
        )

    def forward(self, x):
        logging.debug(f'Decoder inference shape: {x.shape}')
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 10, 5)
        x = self.net(x)
        return x

   
class CNN_decoder(nn.Module):
    def __init__(self, 
                latent_dim: int, 
                input_channels: int = 4,
                channels: list = [32, 16, 8],
                kernel_size: int = 3, 
                ):
        """
        Args:
           latent_dim : Dimensionality of latent representation z
           input_channels : Final number of channels going out from last convolutional layer (always includes cpu_alloc)
           channels : Number of channels we use in the convolutional layers.
        """
        super().__init__()
        modules = []
        
        # input = (N, latent_dim), output = (N, latent_dim * channels[0])
        modules.append(nn.Linear(latent_dim, latent_dim * channels[0])) 
        modules.append(nn.ReLU())
        
        # reshape to (N, channels[0], latent_dim)
        shape_unflattened = (channels[0], latent_dim)
        # input = (N, latent_dim * channels[0]), output = (N, shape_unflattened[0], shape_unflattened[1])
        modules.append(nn.Unflatten(dim=1, unflattened_size=shape_unflattened)) 
        
        # L_out = (L_in - 1) * stride - 2 * padding + dilliation * (kernel_size-1) + output_padding + 1
        modules.append(nn.ConvTranspose1d(channels[0], channels[1], kernel_size=4, padding=3, stride=2))
        modules.append(nn.ReLU())
        
        # L_out = [L_in + 2*padding - dilation*(kernel_size-1) - 1] / stride + 1
        modules.append(nn.Conv1d(channels[1], channels[2], kernel_size=kernel_size, padding='same'))
        modules.append(nn.ReLU())
        
        # L_out = [L_in + 2*padding - dilation*(kernel_size-1) - 1] / stride + 1    
        modules.append(nn.Conv1d(channels[2], input_channels, kernel_size=kernel_size, padding='same'))
        modules.append(nn.ReLU())
        
   
        self.model = nn.Sequential(*modules)
        
        # for channels = [32, 16, 8] and kernel_size = 3 and latent_dim = 32 and input_channels = 4
        # Linear: (N, 32) -> (N, 32 * 32) = (N, 1024)
        # Unflatten: (N, 1024) -> (N, 32, 32)
        # ConvTranspose1d: (N, 32, 32) -> (N, 16, 60)
        # Conv1d: (N, 16, 60) -> (N, 8, 60)
        # Conv1d: (N, 8, 60) -> (N, 4, 60)
    
    def forward(self, x):
        #logging.debug(f'Decoder inference shape: {x.shape}')
        return self.model(x)
    
class CNN_LSTM_decoder(nn.Module):
    def __init__(self, lstm_input_dim: int, 
                 lstm_out_dim: int, 
                 h_lstm_chan: list[int],
                 seq_len: int,
                 input_channels: int=4,) -> None:
        super().__init__()

        """
        Args:
           kernel_size : Kernel size, note that it is applied in Conv1d convolution
           input_channels : Number of channels going into 1st convolutional layer (always includes cpu_alloc)
           lstm_input_dim : Vector length which goes into LSTM block. 
           lstm_out_dim : Final dimension of latent vector before going into CNN block
           h_lstm_chan : middle channels of lstm block
        """

        #LSTM decoder
        layer_dims = [lstm_input_dim] + h_lstm_chan + [h_lstm_chan[-1]]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        self.seq_len = seq_len
        
        for index in range(self.num_layers):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=1,
                batch_first=True
            )
            self.layers.append(layer)

        self.h_activ = nn.Sigmoid()
        self.dense_matrix = nn.Parameter(
            torch.rand((layer_dims[-1], lstm_out_dim), dtype=torch.float),
            requires_grad=True
        )
        
        #CNN decoder
        cnn_modules = []
        channels = [16, 8]

        cnn_modules.append(nn.Linear(40, 80)) #latent_dim 4
        cnn_modules.append(nn.Unflatten(1, (channels[0], 5))) #80 into 16x5
        cnn_modules.append(nn.ConvTranspose1d(channels[0], channels[1], kernel_size=4, stride = 4)) #input (16, 5) output (8,20)
        cnn_modules.append(nn.ReLU())
        cnn_modules.append(nn.ConvTranspose1d(channels[1], input_channels, kernel_size=3, stride =3))
        cnn_modules.append(nn.ReLU())


        self.cnn_decoder = nn.Sequential(*cnn_modules)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.repeat(1, self.seq_len, 1)
        
        for index, layer in enumerate(self.layers):
            x, (h_n, c_n) = layer(x)

            if self.h_activ and index < self.num_layers - 1:
                x = self.h_activ(x)
        
        lstm_decoding = torch.matmul(x, self.dense_matrix)
        cnn_decoding = self.cnn_decoder(lstm_decoding.squeeze(2))
        return cnn_decoding
