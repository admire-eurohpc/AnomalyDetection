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