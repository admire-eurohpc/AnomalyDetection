import logging
import os 
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import configparser

# -- Pytorch imports --
import torch
from torch.utils.data import DataLoader
from torch import optim, nn, utils, Tensor
from torchsummary import summary
import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
import tqdm

# -- Own modules --
from ae_encoder import CNN_encoder
from ae_decoder import CNN_decoder
from ae_litmodel import LitAutoEncoder
from ae_dataloader import TimeSeriesDataset
from utils.plotting import plot_recon_error_agg, plot_recon_error_each_node


logging.basicConfig(level=logging.DEBUG)
config = configparser.ConfigParser()
config.read('config.ini')

_tmp_name = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
logger = TensorBoardLogger(save_dir="lightning_logs", name="ae", version=f'{_tmp_name}')

image_save_path = os.path.join('lightning_logs', 'ae', f'{_tmp_name}')
if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)

SEED = config.getint('TRAINING', 'SEED')
# Setting the seed
L.seed_everything(SEED)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
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
LOGS_PATH = config.get('EVALUATION', 'logs_path')



if __name__ == "__main__":
    
    # Setup data generator class and load it into a pytorch dataloader
    dataset = TimeSeriesDataset(data_dir=f"{PROCESSED_DATA_DIR}/train/", 
                                normalize=True, 
                                window_size=WINDOW_SIZE, 
                                slide_length=TRAIN_SLIDE)
    train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        drop_last=True,
        **kwargs,
        )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=BATCH_SIZE,
        shuffle=VAL_SHUFFLE,
        drop_last=True,
        **kwargs,
        )
    
    # Setup test data
    test_dataset = TimeSeriesDataset(data_dir=f"{PROCESSED_DATA_DIR}/test/", 
                                     normalize=True, 
                                     external_transform=dataset.get_transform(), # Use same transform as for training
                                     window_size=WINDOW_SIZE, 
                                     slide_length=TEST_SLIDE)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, **kwargs)
    
    # Get input size and shapes
    d = next(iter(test_dataloader))
    input_shape = d.shape
    channels = d.shape[0]
    width = d.shape[1]
    test_len = len(test_dataset)
    logging.debug(f"test input_shape: {test_len}")

    logging.debug(f"input_shape: {input_shape}")
    logging.debug(f'channels: {channels}, width: {width}')
    
    # Log hyperparameters for tensorboard
    logger.log_hyperparams({
        'batch_size': BATCH_SIZE,
        'max_epochs': MAX_EPOCHS,
        'shuffle': SHUFFLE,
        'window_size': WINDOW_SIZE,
        'train_slide': TRAIN_SLIDE,
        'test_slide': TEST_SLIDE,
        'latent_dim': LATENT_DIM,
        'number_of_channels': channels,
        'seed': SEED,
    })

    # Init the lightning autoencoder
    cnn_encoder = CNN_encoder(kernel_size=10, latent_dim=LATENT_DIM, cpu_alloc=INCLUDE_CPU_ALLOC)
    cnn_decoder = CNN_decoder(latent_dim=LATENT_DIM, cpu_alloc=INCLUDE_CPU_ALLOC)
    autoencoder = LitAutoEncoder(input_shape, LATENT_DIM, cnn_encoder, cnn_decoder)

    # Add early stopping
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", 
        min_delta=0.00, 
        patience=10, 
        verbose=False, 
        mode="min"
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=config.getint('TRAINING', 'SAVE_TOP_K'),
        verbose=True, 
        monitor="val_loss", 
        mode="min"
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval = 'epoch')


    logging.debug(f'Autoencoder Summary: {autoencoder}')
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        logger=logger,
        callbacks=[
            early_stop_callback,
            checkpoint_callback,
            lr_callback,
            ],   
        enable_checkpointing=config.getboolean('TRAINING', 'ENABLE_CHECKPOINTING'),
        accelerator="gpu", devices=1, strategy="auto",
        )
    trainer.fit(
        model=autoencoder, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
        )

    logging.debug(os.path.join(logger.save_dir, logger.name, logger.version, "checkpoints"))
    
    autoencoder = LitAutoEncoder.load_from_checkpoint(
                                checkpoint_path=checkpoint_callback.best_model_path,
                                encoder=cnn_encoder, 
                                decoder=cnn_decoder,
                                input_shape=input_shape,
                                latent_dim=LATENT_DIM,
                                map_location="cuda:0"
                                )
    # ----- TEST ----- #

    # Now test the model on february data
    # Run the model on the entire test set and report reconstruction error to tensorboard
    autoencoder.eval()
    autoencoder.freeze()

    logging.debug(f"Running model on test set")

    test_recon_mae = []
    node_len = test_dataset.get_node_len()
    full_node_len = test_dataset.get_node_full_len()

    # Run evaluation on test set
    for idx, batch in tqdm.tqdm(enumerate(test_dataloader), desc="Running test reconstruction error", total=test_len):
        batch = batch.to(device)
        err = torch.mean(torch.abs(batch - autoencoder.decoder(autoencoder.encoder(batch))))
        err_detached = err.cpu().numpy()
        test_recon_mae.append(err_detached)

    '''Due to processing whole test set at once (one node after one, last WINDOW_SIZE-1 samples of node x
    are processed together with samples from x+1 node resulting in high reconstruction errors on the beginning and the end of test_set)
    On training set this is also present but considering vast amount of samples in each node the impact is insignificant'''

    test_recon_mae_stripped = []
    for i in range(NODES_COUNT):
        test_recon_mae_stripped.append(test_recon_mae[(i*full_node_len): (node_len*(i+1)+(full_node_len-node_len)*i)])

    test_recon_mae_np = np.reshape(test_recon_mae_stripped, (NODES_COUNT, node_len))
    test_recon_mae_list = test_recon_mae_np.tolist()
    agg_recon_err = np.mean(test_recon_mae_np, axis=0)

    test_date_range = test_dataset.get_dates_range()
    test_date_range = pd.date_range(start=test_date_range['start'], end=test_date_range['end'], freq=f'{TEST_SLIDE}min', tz='Europe/Warsaw') # TODO set frequency dynamically?

    logging.debug(f'Dates range test: start {test_date_range.min()} end {test_date_range.max()}')
    logging.debug(f'Test len: {len(test_dataloader)}, Dates len: {len(test_date_range)}, Recon len: {len(test_recon_mae)}')

    hostnames = test_dataset.get_filenames()

    # Display the reconstruction error over time manually
    logging.debug(f"Plotting reconstruction error over time")

    plot_recon_error_each_node(reconstruction_errors = test_recon_mae_list, time_axis = test_date_range, n_nodes = 200, hostnames = hostnames, savedir=LOGS_PATH)
    plot_recon_error_agg(reconstruction_errors = agg_recon_err, time_axis = test_date_range, hostnames = 'mean recon_error', savedir=LOGS_PATH)


    # Plot some reconstructions vs real examples
    # logging.debug(f"Plotting reconstructions vs real")
    # sample = test_dataset.get_time_series()
    # sample = torch.Tensor(sample[:, :, 0:WINDOW_SIZE].flatten())
    # sample = sample.to(device)
    # reconstructions = autoencoder.decoder(autoencoder.encoder(sample)).cpu().numpy()
    # sample = sample.cpu().numpy()
    # plot_embeddings_vs_real(
    #     _embeddings=reconstructions,
    #     _real=sample,
    #     channels=channels,
    #     height=height,
    #     width=width,
    #     checkpoint=checkpoint_callback.best_model_path,
    #     image_save_path=image_save_path,
    #     write=True,
    #     show=False,
    # )


    logging.debug(f"Finished")
    
