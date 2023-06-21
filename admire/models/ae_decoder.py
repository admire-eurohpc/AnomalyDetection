import logging
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
    def __init__(self, latent_dim: int, cpu_alloc: bool):
        """
        Args:
           num_input_channels : Number of input channels 3 without cpus_alloc, 4 with this feature
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        modules = []
        channels = [800, 400, 200]

        if cpu_alloc:
            kernel_size2d = (4,3)
        else:
            kernel_size2d = 3
        modules.append(nn.Linear(latent_dim, latent_dim*100)) #potem reshape 60 na 6x10
        modules.append(nn.Unflatten(1, (channels[0], 5))) #4000 into 800x5
        modules.append(nn.ConvTranspose1d(channels[0], channels[1], kernel_size=4, stride = 4)) #input (800, 5) output (800,20)
        modules.append(nn.Unflatten(2, (1, 20)))
        modules.append(nn.ConvTranspose2d(channels[1], channels[2], kernel_size=kernel_size2d, stride =3))
        modules.append(nn.ReLU())


        self.model = nn.Sequential(*modules)
    
    def forward(self, x):
        #logging.debug(f'Decoder inference shape: {x.shape}')
        return self.model(x)
   
class CNN_decoder_one_node(nn.Module):
    def __init__(self, latent_dim: int, cpu_alloc: bool):
        """
        Args:
           num_input_channels : Number of input channels 3 without cpus_alloc, 4 with this feature
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        modules = []
        channels = [32, 8, 1]

        if cpu_alloc:
            kernel_size2d = (4,3)
        else:
            kernel_size2d = 3
        modules.append(nn.Linear(latent_dim, latent_dim*40)) #latent_dim 4
        modules.append(nn.Unflatten(1, (channels[0], 5))) #160 into 32x5
        modules.append(nn.ConvTranspose1d(channels[0], channels[1], kernel_size=4, stride = 4)) #input (32, 5) output (32,20)
        modules.append(nn.Unflatten(2, (1, 20)))
        modules.append(nn.ConvTranspose2d(channels[1], channels[2], kernel_size=kernel_size2d, stride =3))
        modules.append(nn.ReLU())


        self.model = nn.Sequential(*modules)
    
    def forward(self, x):
        #logging.debug(f'Decoder inference shape: {x.shape}')
        return self.model(x)
