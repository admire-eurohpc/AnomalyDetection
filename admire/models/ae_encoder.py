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