import logging
import torch.nn as nn

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
    def __init__(self, kernel_size: int, latent_dim: int, cpu_alloc: bool):
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
        channels = [200, 400, 800]
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
        modules.append(nn.Linear(latent_dim*100, latent_dim)) #latent dim of 12-6 preferably

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        #logging.debug(f'Encoder inference shape: {x.shape}')
        return self.model(x)


class CNN_encoder_one_node(nn.Module):
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