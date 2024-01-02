import os
from typing import Any
from torch import optim, nn, utils, Tensor
from torchmetrics import MeanAbsoluteError, R2Score, MeanAbsolutePercentageError
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import logging

# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, 
                encoder: nn.Module,
                decoder: nn.Module,
                lr: int = 1e-4,
                monitor: str = 'train_loss',
                monitor_mode: str = 'min',
                ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr
        self.monitor = monitor
        self.monitor_mode = monitor_mode

        self.mae = MeanAbsoluteError()
        self.mape = MeanAbsolutePercentageError()
        
        
        
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def _get_reconstruction_loss(self, x, x_hat):
        loss = F.mse_loss(x, x_hat, reduction='none')
        loss = loss.sum(dim=1).mean()
        return loss

    def _get_reconstruction_mae(self, x, x_hat):
        mae = self.mae(x_hat, x)
        return mae

    def _get_reconstruction_mape(self, x, x_hat):
        mape = self.mae(x_hat, x)
        return mape
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), 
                                lr=self.lr, 
                            )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                               factor=0.5, 
                                                               patience=5,
        )
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': self.monitor,
                'mode': self.monitor_mode,
                }
    
    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self.forward(x)
        loss = self._get_reconstruction_loss(x, x_hat)
        self.log('train_loss', loss)
        self.log('train_mae', self._get_reconstruction_mae(x, x_hat))
        self.log('train_mape', self._get_reconstruction_mape(x, x_hat))
        
        self.log_gradients_in_model(self.global_step)
        
        return loss
    
    
    def test_step(self, batch, batch_idx):
        x = batch
        x_hat = self.forward(x)
        loss = self._get_reconstruction_loss(x, x_hat)
        self.log('test_loss(mse)', loss)
        self.log('test_mae', self._get_reconstruction_mae(x, x_hat))
        self.log('test_mape', self._get_reconstruction_mape(x, x_hat))


    def log_gradients_in_model(self, step):
        for tag, value in self.named_parameters():
            if value.grad is not None:
                
                self.tensorboard_logger = self.trainer.logger.experiment
                self.tensorboard_logger.add_histogram(tag + "/grad", value.grad.cpu(), step)
         

class LSTMAE_Encoder(nn.Module):
    def __init__(self, window_size, hidden_size, latent_size, device, channels=4) -> None:
        
        super().__init__()
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.device = device
        
        self.cnn1d = nn.Conv1d(in_channels=channels,
                               out_channels=1, 
                               kernel_size=1, 
                               padding='same',
        )
        
        self.flatten = nn.Flatten()
        
        self.lstm = nn.LSTM(input_size=window_size, 
                            hidden_size=hidden_size, 
                            num_layers=1, 
                            batch_first=True,
                            bidirectional=False,
                            )
        
        self.linear = nn.Linear(hidden_size, latent_size)
    
        
    def forward(self, x):
        x = self.cnn1d(x)
        x = self.flatten(x)
        outputs, (hidden, cell) = self.lstm(x)
        outputs = self.linear(outputs)
        return outputs
    
class LSTMAE_Decoder(nn.Module):
    def __init__(self, window_size, hidden_size, latent_size, device, channels=4) -> None:
        
        super().__init__()
        self.window_size = window_size
        self.channels = channels
    
        self.lstm = nn.LSTM(input_size=latent_size, 
                            hidden_size=hidden_size, 
                            num_layers=1, 
                            batch_first=True,
                            bidirectional=False,
                            )
        
        self.linear = nn.Linear(hidden_size, window_size)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(window_size, window_size * channels)
        
    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        outputs = self.linear(outputs)
        outputs = self.relu(outputs)
        outputs = self.linear1(outputs)
        outputs = outputs.view(-1, self.channels, self.window_size)
        return outputs
       
class LSTM_AE(pl.LightningModule):
    
    def __init__(self, window_size, hidden_size, latent_size, device, lr, channels = 4,  monitor: str = 'train_loss',
                monitor_mode: str = 'min',) -> None:
        
        super().__init__()
        
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.channels = channels
        
        self.lr = lr
        self.mae = MeanAbsoluteError()
        self.mape = MeanAbsolutePercentageError()
        
        self.encoder = LSTMAE_Encoder(window_size=window_size, 
                                      hidden_size=hidden_size, 
                                      latent_size=latent_size, 
                                      device=device,
                                      channels=channels,
        )
    
        self.decoder = LSTMAE_Decoder(window_size=window_size,
                                    hidden_size=hidden_size, 
                                    latent_size=latent_size, 
                                    device=device,
                                    channels=channels,
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, x):
        outputs = self.encoder(x)
        outputs = self.decoder(outputs)
        return outputs
        
    def _get_reconstruction_loss(self, x, x_hat):
        loss = F.mse_loss(x, x_hat, reduction='none')
        loss = loss.sum(dim=1).mean()
        return loss

    def _get_reconstruction_mae(self, x, x_hat):
        mae = self.mae(x_hat, x)
        return mae

    def _get_reconstruction_mape(self, x, x_hat):
        mape = self.mae(x_hat, x)
        return mape
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), 
                                lr=self.lr, 
                            )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                               factor=0.5, 
                                                               patience=5,
        )
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': self.monitor,
                'mode': self.monitor_mode,
                }
    
    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self.forward(x)
        loss = self._get_reconstruction_loss(x, x_hat)
        self.log('train_loss', loss)
        self.log('train_mae', self._get_reconstruction_mae(x, x_hat))
        self.log('train_mape', self._get_reconstruction_mape(x, x_hat))
        
        self.log_gradients_in_model(self.global_step)
        
        return loss
    
    
    def test_step(self, batch, batch_idx):
        x = batch
        x_hat = self.forward(x)
        loss = self._get_reconstruction_loss(x, x_hat)
        self.log('test_loss(mse)', loss)
        self.log('test_mae', self._get_reconstruction_mae(x, x_hat))
        self.log('test_mape', self._get_reconstruction_mape(x, x_hat))


    def log_gradients_in_model(self, step):
        for tag, value in self.named_parameters():
            if value.grad is not None:
                
                self.tensorboard_logger = self.trainer.logger.experiment
                self.tensorboard_logger.add_histogram(tag + "/grad", value.grad.cpu(), step)