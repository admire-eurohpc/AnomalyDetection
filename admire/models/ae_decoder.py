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
    def __init__(self, 
                latent_dim: int, 
                cpu_alloc: bool,
                channels: list = [32, 16, 8],
                kernel_size: int = 3, 
                ):
        """
        Args:
           num_input_channels : Number of input channels 3 without cpus_alloc, 4 with this feature
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        modules = []

        if cpu_alloc: input_channels = 4
        else: input_channels = 3
        
        # input = (N, latent_dim), output = (N, latent_dim * channels[0])
        modules.append(nn.Linear(latent_dim, latent_dim * channels[0])) 
        modules.append(nn.ELU())
        
        # reshape to (N, channels[0], latent_dim)
        shape_unflattened = (channels[0], latent_dim)
        # input = (N, latent_dim * channels[0]), output = (N, shape_unflattened[0], shape_unflattened[1])
        modules.append(nn.Unflatten(dim=1, unflattened_size=shape_unflattened)) 
        
        # L_out = (L_in - 1) * stride - 2 * padding + dilliation * (kernel_size-1) + output_padding + 1
        modules.append(nn.ConvTranspose1d(channels[0], channels[1], kernel_size=4, padding=3, stride=2))
        modules.append(nn.ELU())
        
        # L_out = [L_in + 2*padding - dilation*(kernel_size-1) - 1] / stride + 1
        modules.append(nn.Conv1d(channels[1], channels[2], kernel_size=kernel_size, padding='same'))
        modules.append(nn.ELU())
        
        # L_out = [L_in + 2*padding - dilation*(kernel_size-1) - 1] / stride + 1    
        modules.append(nn.Conv1d(channels[2], input_channels, kernel_size=kernel_size, padding='same'))
        modules.append(nn.ELU())
        
   
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
