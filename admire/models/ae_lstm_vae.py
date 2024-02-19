from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
from torch import nn 
from torch.nn import functional as F 
import lightning as L 

class LSTMEncoder(nn.Module):
    def __init__(self, 
                 channels: int,
                 hidden_size: int, 
                 embedding_filters: int,
                 num_layers: int = 2,
        ):
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.channels = channels
        
        self.cnn1d = nn.Conv1d(
            in_channels=channels,
            out_channels=embedding_filters,
            kernel_size=1,
            padding='same',
        )
        self.relu = nn.LeakyReLU()
        self.flatten = nn.Flatten()
        
        # self.fc1 = nn.Linear(input_size, input_size * hidden_size)
        
        self.lstm = nn.LSTM(
            embedding_filters,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
            bias=True,
        )

    def forward(self, x):
        # x: tensor of shape (batch_size, channels, seq_length)
        _x = self.cnn1d(x) # Embeding to batch_size x embedding_filters x seq_length
        _x = self.relu(_x)
        _x = _x.permute(0, 2, 1) # swap the dimensions for LSTM
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        outputs, (hidden, cell) = self.lstm(_x) # LSTM 
        return (hidden, cell)


class LSTMDecoder(nn.Module):
    def __init__(
        self, 
        latent_size: int, 
        channels: int,
        hidden_size: int, 
        output_size: int, 
        num_layers: int = 2,
    ):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size * channels
        self.num_layers = num_layers
        self.channels = channels
        
        self.lstm = nn.LSTM(
            latent_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
        
        
        self.relu = nn.LeakyReLU()
        
        self.cov1d = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=channels,
            kernel_size=1
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        output, (hidden, cell) = self.lstm(x)
        xhat = self.relu(output)
        
        # swap the dimensions for Batch x Channels x Seq_length
        xhat = xhat.permute(0, 2, 1)
        xhat = self.cov1d(xhat)
        
        xhat = self.sigmoid(xhat)
    
        return xhat, (hidden, cell)
    
    
class LSTMVAE(L.LightningModule):
    """LSTM-based Variational Auto Encoder"""

    def __init__(
        self, 
        input_size: int, 
        channels: int,
        hidden_size: int = 64, 
        latent_size: int = 16,
        num_lstm_layers: int = 2, 
        embedding_filters: int = 16,
        lr: float = 2e-4,
        kld_weight: float = 1.0,
        beta: float = 10.0,
        kl_weighting_scheme: str = 'Normal',
        monitor: str = 'train_loss',
        monitor_mode: str = 'min',
        ):
        """
        input_size: int, input_dim
        channels: int, number of channels in input
        hidden_size: int, output size of LSTM AE
        latent_size: int, latent z-layer size
        num_lstm_layer: int, number of layers in LSTM
        embedding_filters: int, number of filters in CNN
        lr: int, learning rate
        kld_weight: float, weight of KLD loss, default 1.0
        beta: float, beta value for B-Norm, beta should be > 1.0
        kl_weighting_scheme: str, KLD weighting scheme, default B-Norm, options: B-Norm, Normal
        monitor: str, metric to monitor, default train_loss
        monitor_mode: str, mode of monitoring, default min
        """
        super(LSTMVAE, self).__init__()
        
        self.lr = lr
        self.kld_weight_normal = kld_weight
        self.kld_weight_B_norm = (beta * latent_size) / input_size #https://openreview.net/pdf?id=Sy2fzU9gl
        
        self.kld_weight = self.kld_weight_normal if kl_weighting_scheme == 'Normal' else self.kld_weight_B_norm
        self.log('kld_weight', self.kld_weight)
        
        self.monitor = monitor
        self.monitor_mode = monitor_mode

        # dimensions
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_lstm_layers

        # lstm ae
        self.encoder = LSTMEncoder(
            channels=channels,
            hidden_size=hidden_size, 
            embedding_filters=embedding_filters,
            num_layers=self.num_layers
        )
        self.decoder = LSTMDecoder(
            latent_size=latent_size,
            channels=channels,
            output_size=input_size,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
        )

        self.fc21 = nn.Linear(self.hidden_size, self.latent_size)
        self.fc22 = nn.Linear(self.hidden_size, self.latent_size)
        self.fc3 = nn.Linear(self.latent_size, self.hidden_size)
        
        
        self.save_hyperparameters()

    def reparametize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std).to(self.device)

        z = mu + noise * std
        return z

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        ) * self.kld_weight

        loss = recons_loss + kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD_Loss": kld_loss.detach(),
        }

    def forward(self, x):
        batch_size, feature_dim, seq_len, = x.shape

        # encode input space to hidden space
        enc_hidden, _ = self.encoder(x)
        enc_h = enc_hidden[-1].view(batch_size, self.hidden_size).to(self.device)

        # extract latent variable z(hidden space to latent space)
        mean = self.fc21(enc_h)
        logvar = self.fc22(enc_h)
        z = self.reparametize(mean, logvar)  # batch_size x latent_size

        # initialize hidden state as inputs
        h_ = self.fc3(z)
        
        # decode latent space to input space
        # z = z.repeat(1, seq_len, 1)
        # z = z.view(batch_size, seq_len, self.latent_size).to(self.device)
        
        # z is B X L 
        # _h is B x H
        # we need z to be B x L x H
        z = z.unsqueeze(1).repeat(1, seq_len, 1)
        z = z.view(batch_size, seq_len, self.latent_size).to(self.device)

        # initialize hidden state
        # hidden = (h_.contiguous(), h_.contiguous())
        reconstruct_output, hidden = self.decoder(z)

        return reconstruct_output, (mean, logvar)

    def training_step(self, batch, batch_idx):
        return self.step('train', batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.step('val', batch, batch_idx)
    
    def test_step(self, batch, batch_idx):
        return self.step('test', batch, batch_idx)
    
    def step(self, _type, batch, batch_idx):
        x_hat, (mean, logvar) = self.forward(batch)
        
        losses = self.loss_function(x_hat, batch, mean, logvar)
        m_loss, recon_loss, kld_loss = (
            losses["loss"],
            losses["Reconstruction_Loss"],
            losses["KLD_Loss"],
        )
    
        self.log_losses(_type, losses)
        
        return m_loss
    
    def log_losses(self, _type: str, losses: dict):
        for key, value in losses.items():
            self.log(f'{_type}_{key}', value)
              
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            factor=0.5, 
            patience=5,
            min_lr=self.lr / 3
        )
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': self.monitor,
                'mode': self.monitor_mode,
        }

