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
                 input_shape: tuple,
                 latent_dim: int,
                 encoder: nn.Module,
                 decoder: nn.Module
                 ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        #self.example_input_array = torch.zeros(2, *input_shape)
        
    def forward(self, x):
        #logging.info(x.shape)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def _get_reconstruction_loss(self, x, x_hat):
        loss = F.mse_loss(x, x_hat, reduction='none')
        loss = loss.sum(dim=1).mean()
        return loss

    def _get_reconstruction_mae(self, x, x_hat):
        mae = MeanAbsoluteError()(x_hat, x)
        return mae

    def _get_reconstruction_mape(self, x, x_hat):
        mape = MeanAbsolutePercentageError()(x_hat, x)
        return mape
    
    def _get_reconstruction_r2(self, x, x_hat):
        logging.info(f'Validation: {x.shape} vs {x_hat.shape}')
        r2 = R2Score(adjusted=True)(x_hat, x)
        return r2
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self.forward(x)
        loss = self._get_reconstruction_loss(x, x_hat)
        self.log('train_loss', loss)
        self.log('train_mae', self._get_reconstruction_mae(x, x_hat))
        self.log('train_mape', self._get_reconstruction_mape(x, x_hat))
        #self.log('train_r2', self._get_reconstruction_r2(x, x_hat))
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
        self.log('test_loss', loss)
        self.log('test_mae', self._get_reconstruction_mae(x, x_hat))

