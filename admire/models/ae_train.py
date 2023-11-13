import logging
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import configparser
import tqdm

# -- Pytorch imports --
import torch
from torch.utils.data import DataLoader
from torch import optim, nn, utils, Tensor
from torchsummary import summary
import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.accelerators import CPUAccelerator

# -- Other Imports --
from scipy.stats import zscore
from datetime import datetime


# -- Own modules --
from ae_encoder import CNN_encoder, CNN_LSTM_encoder
from ae_decoder import CNN_decoder, CNN_LSTM_decoder
from ae_litmodel import LitAutoEncoder
from ae_dataloader import TimeSeriesDataset
from utils.plotting import plot_recon_error_agg, plot_recon_error_each_node


logging.basicConfig(level=logging.DEBUG)
config = configparser.ConfigParser()
config.read('config.ini')

_tmp_name = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
tensorboard_logging_path = config.get('TRAINING', 'tensorboard_logging_path')
img_save_dirname = config.get('TRAINING', 'image_save_path')

logger = TensorBoardLogger(save_dir=tensorboard_logging_path, name="AE_CNN_ONE_NODE", version=f'{_tmp_name}')
image_save_path = os.path.join(logger.log_dir, img_save_dirname)

if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)

SEED = config.getint('TRAINING', 'SEED')
# Setting the seed
L.seed_everything(SEED)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

BATCH_SIZE = config.getint('TRAINING', 'BATCH_SIZE')
MAX_EPOCHS = config.getint('TRAINING', 'MAX_EPOCHS')
WINDOW_SIZE = config.getint('TRAINING', 'WINDOW_SIZE')
TRAIN_SLIDE = config.getint('TRAINING', 'TRAIN_SLIDE')
TEST_SLIDE = config.getint('TRAINING', 'TEST_SLIDE')
LATENT_DIM = config.getint('TRAINING', 'LATENT_DIM')
NODES_COUNT = config.getint('PREPROCESSING', 'nodes_count_to_process')
LR = config.getfloat('TRAINING', 'LEARNING_RATE')
SHUFFLE = config.getboolean('TRAINING', 'SHUFFLE')
VAL_SHUFFLE = config.getboolean('TRAINING', 'VAL_SHUFFLE')
INCLUDE_CPU_ALLOC = config.getboolean('PREPROCESSING', 'with_cpu_alloc')
PROCESSED_DATA_DIR = config.get('PREPROCESSING', 'processed_data_dir')
ENCODER_LAYERS = config.get('TRAINING', 'ENCODER_LAYERS')
DECODER_LAYERS = config.get('TRAINING', 'DECODER_LAYERS')


if __name__ == "__main__":
    
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    accelerator = 'gpu' if use_cuda else 'cpu'
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {'num_workers': CPUAccelerator().auto_device_count(), 'pin_memory': False}
    

    # Setup data generator class and load it into a pytorch dataloader
    dataset = TimeSeriesDataset(data_dir=f"{PROCESSED_DATA_DIR}/train/", 
                                normalize=True, 
                                window_size=WINDOW_SIZE, 
                                slide_length=TRAIN_SLIDE)
    
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        drop_last=True,
        **kwargs,
        )
    
    # Get input size and shapes
    d = next(iter(train_loader))
    input_shape = d.shape
    batch = d.shape[0]
    n_features = d.shape[1]
    win_size = d.shape[2]

    logging.debug(f"input_shape: {input_shape}")
    logging.debug(f'batch: {batch}, number of features: {n_features}, window size: {win_size}')

    
    # Log hyperparameters for tensorboard
    logger.log_hyperparams({
        'batch_size': BATCH_SIZE,
        'max_epochs': MAX_EPOCHS,
        'shuffle': SHUFFLE,
        'window_size': WINDOW_SIZE,
        'train_slide': TRAIN_SLIDE,
        'test_slide': TEST_SLIDE,
        'latent_dim': LATENT_DIM,
        'encoder_layers': ENCODER_LAYERS,
        'decoder_layers': DECODER_LAYERS,
        'seed': SEED,
    })
    

    # Init the lightning autoencoder
    cnn_encoder = CNN_encoder(kernel_size=3, latent_dim=LATENT_DIM, cpu_alloc=INCLUDE_CPU_ALLOC, channels=eval(ENCODER_LAYERS))
    cnn_decoder = CNN_decoder(kernel_size=3, latent_dim=LATENT_DIM, cpu_alloc=INCLUDE_CPU_ALLOC, channels=eval(DECODER_LAYERS))
    autoencoder = LitAutoEncoder(cnn_encoder, cnn_decoder, monitor='train_loss', monitor_mode='min')

    # LSTM
    # cnn_lstm_encoder = CNN_LSTM_encoder(lstm_input_dim=1, lstm_out_dim=48, h_lstm_chan=[96], cpu_alloc=True)
    # cnn_lstm_decoder = CNN_LSTM_decoder(lstm_input_dim=48, lstm_out_dim =1, h_lstm_chan=[96], cpu_alloc=True)
    # lstm_conv_autoencoder = LitAutoEncoder(cnn_lstm_encoder, cnn_lstm_decoder)

    # Add early stopping
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="train_loss", 
        min_delta=0.001, 
        patience=5, 
        verbose=True, 
        mode="min"
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=config.getint('TRAINING', 'SAVE_TOP_K'),
        verbose=True, 
        monitor="train_loss", 
        mode="min"
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval = 'epoch')

    profiler = SimpleProfiler()

    logging.debug(f'Autoencoder Summary: {autoencoder}')
    # logging.debug(f'Autoencoder Summary: {lstm_conv_autoencoder}')
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        logger=logger,
        callbacks=[
            early_stop_callback,
            checkpoint_callback,
            lr_callback,
            ],   
        enable_checkpointing=config.getboolean('TRAINING', 'ENABLE_CHECKPOINTING'),
        accelerator=accelerator,
        devices="auto", 
        strategy="auto",
        profiler=profiler,
        )
    trainer.fit(
        model=autoencoder, 
        train_dataloaders=train_loader
        # LSTM
        # model=lstm_conv_autoencoder, 
        # train_dataloaders=train_loader,
        )

    logging.debug(os.path.join(logger.save_dir, logger.name, logger.version, "checkpoints"))
    