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
    def __init__(self, 
                kernel_size: int = 3, 
                latent_dim: int = 32, 
                cpu_alloc: bool = False,
                window_size: int = 60,
                batch_size: int = 1,
                channels: list = [16, 32, 64],
                ):
        """
        Args:
           num_input_channels : Number of input channels 3 without cpus_alloc, 4 with this feature
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        
        if cpu_alloc: input_channels = 4
        else: input_channels = 3
        
        modules = []
        modules.append(nn.Conv1d(input_channels, channels[0], kernel_size=kernel_size, padding='same')) #input = (N x 4 x 60), output = (N x channel[0] x 60)
        modules.append(nn.ELU())
        modules.append(nn.AvgPool1d(3)) # input = (N x channel[0] x 60), output = (N x channel[0] x 20)
        modules.append(nn.Dropout(0.2))
        
        modules.append(nn.Conv1d(channels[0], channels[1], kernel_size=kernel_size, padding='same'))  # input = (N x channel[0] x 20), output = (N x channel[1] x 20)
        modules.append(nn.ELU())
        modules.append(nn.AvgPool1d(3)) # input = (N x channel[1] x 20), output = (N x channel[1] x 6)
        modules.append(nn.Dropout(0.2))
        
        modules.append(nn.Conv1d(channels[1], channels[2], kernel_size=kernel_size, padding='same')) # input = (N x channel[1] x 6), output = (N x channel[2] x 6)
        modules.append(nn.ELU())
        modules.append(nn.AvgPool1d(3)) # input = (N x channel[2] x 6), output = (N x channel[2] x 2)
        modules.append(nn.Dropout(0.2))
        
        modules.append(nn.Flatten(start_dim=1)) # input = (N x channel[2] x 2), output = (N x channel[2] * 2)
        
        modules.append(nn.Linear(channels[-1] * 2, latent_dim)) # latent space = (N x latent_dim)

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        # logging.debug(f'Encoder inference shape: {x.shape}')
        return self.model(x)