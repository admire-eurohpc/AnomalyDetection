import os
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
        self.log('test_reconstruction_loss(mse)', loss)
        self.log('test_reconstruction_mae', self._get_reconstruction_mae(x, x_hat))


    def log_gradients_in_model(self, step):
        for tag, value in self.named_parameters():
            if value.grad is not None:
                
                self.tensorboard_logger = self.trainer.logger.experiment
                self.tensorboard_logger.add_histogram(tag + "/grad", value.grad.cpu(), step)