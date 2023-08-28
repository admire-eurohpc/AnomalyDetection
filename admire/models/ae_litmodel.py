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
                 ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr

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
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_loss',
                }
    
    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self.forward(x)
        loss = self._get_reconstruction_loss(x, x_hat)
        self.log('train_loss', loss)
        self.log('train_mae', self._get_reconstruction_mae(x, x_hat))
        self.log('train_mape', self._get_reconstruction_mape(x, x_hat))
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self.forward(x)
        loss = self._get_reconstruction_loss(x, x_hat)
        self.log('val_loss', loss)
        self.log('val_mae', self._get_reconstruction_mae(x, x_hat))
        self.log('val_mape', self._get_reconstruction_mape(x, x_hat))
        #self.log('val_r2', self._get_reconstruction_r2(x, x_hat))
    
    def test_step(self, batch, batch_idx):
        x = batch
        x_hat = self.forward(x)
        loss = self._get_reconstruction_loss(x, x_hat)
        self.log('test_reconstruction_loss(mse)', loss)
        self.log('test_reconstruction_mae', self._get_reconstruction_mae(x, x_hat))

