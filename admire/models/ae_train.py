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
from utils.plotting import plot_embeddings_vs_real, plot_reconstruction_error_over_time, plot_recon_error_each_node, plot_recon_error_each_node_df


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

ENCODER_LAYERS = np.array([int(x) for x in config.get('TRAINING', 'ENCODER_LAYERS').split(',')])
DECODER_LAYERS = np.array([int(x) for x in config.get('TRAINING', 'DECODER_LAYERS').split(',')])
LATENT_DIM = config.getint('TRAINING', 'LATENT_DIM')
LR = config.getfloat('TRAINING', 'LEARNING_RATE')
SHUFFLE = config.getboolean('TRAINING', 'SHUFFLE')
VAL_SHUFFLE = config.getboolean('TRAINING', 'VAL_SHUFFLE')

PROCESSED_DATA_DIR = config.get('PREPROCESSING', 'processed_data_dir')
INCLUDE_CPU_ALLOC = config.getboolean('PREPROCESSING', 'with_cpu_alloc')

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
        'encoder_layers': ENCODER_LAYERS,
        'decoder_layers': DECODER_LAYERS,
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
    # # ----- TEST ----- #
    # # Now test the model on february data
    # # Run the model on the entire test set and report reconstruction error to tensorboard
    autoencoder.eval()
    autoencoder.freeze()
    
    logging.debug(f"Running model on test set")

    test_reconstruction_mean_absolute_error = []
    # Run evaluation on test set
    for idx, batch in tqdm.tqdm(enumerate(test_dataloader), desc="Running test reconstruction error", total=test_len):
        batch = batch.to(device)
        err = torch.mean(torch.abs(batch - autoencoder.decoder(autoencoder.encoder(batch))))
        err_detached = err.cpu().numpy()
        logger.experiment.add_scalar("test_reconstruction_error", err_detached, idx)
        test_reconstruction_mean_absolute_error.append(err_detached)
    
    cut = test_len%(200)
    test_res = np.reshape(test_reconstruction_mean_absolute_error[0:-cut], (200, test_len//200))
    test_res = test_res.tolist()
    # Plot reconstruction error over time
    dates_range = test_dataset.get_dates_range()
    hostnames = test_dataset.get_filenames()
    #df = pd.DataFrame(data={'ReconError': test_res, 'hostnames': hostnames})
    logging.debug(f'Date range: {dates_range["end"]} - {dates_range["start"]}')
    dates_range = pd.date_range(start=dates_range["start"], end=dates_range["end"], freq='1min', tz='Europe/Warsaw') #shift of 2 hours
    dates_range = dates_range.to_numpy()[:len(test_reconstruction_mean_absolute_error)] # Fit dates range to actual data (bear in mind that last date is max - WINDOW_SIZE)
    logging.debug(f"Plotting reconstruction error over time")
    plot_recon_error_each_node(n_nodes = 200, time_axis = dates_range, data = test_res, hostnames = hostnames, savedir=image_save_path)
    # plot_reconstruction_error_over_time(
    #     reconstruction_errors=test_reconstruction_mean_absolute_error,
    #     time_axis=dates_range,
    #     write=True,
    #     show=False,
    #     savedir=image_save_path
    # )

    # Matplotlib scatter
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(dates_range, test_reconstruction_mean_absolute_error)
    ax.set_xlabel('Date')
    ax.set_ylabel('Reconstruction error')
    ax.set_title('Reconstruction error over time')
    fig.savefig(os.path.join(image_save_path, 'plt_reconstruction_error_over_time.png'))
    logger.experiment.add_figure("reconstruction_error_over_time_figure", fig)
    logging.debug(f"Saved reconstruction error over time figure")


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
    
