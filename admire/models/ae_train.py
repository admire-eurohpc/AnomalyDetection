import logging
import os 
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# -- Pytorch imports --
import torch
from torch.utils.data import DataLoader
from torch import optim, nn, utils, Tensor
import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
import tqdm

# -- Own modules --
from ae_encoder import Encoder
from ae_decoder import Decoder
from ae_litmodel import LitAutoEncoder
from ae_dataloader import TimeSeriesDataset
from utils.plotting import plot_embeddings_vs_real, plot_reconstruction_error_over_time


logging.basicConfig(level=logging.DEBUG)


image_save_path = os.path.join('images', 'training')
if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)


_tmp_name = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
logger = TensorBoardLogger(save_dir="lightning_logs", name="ae", version=f'{_tmp_name}')


# Setting the seed
L.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


BATCH_SIZE = 32
MAX_EPOCHS = 1
SHUFFLE = True
VAL_SHUFFLE = False
WINDOW_SIZE = 20
TRAIN_SLIDE = 10
TEST_SLIDE = 1


# THIS NUMBERS WILL BE MULTIPLIED BY NUMBER OF NODES 
ENCODER_LAYERS = np.array([24, 12])
DECODER_LAYERS = np.array([12, 24])
LATENT_DIM = 6 


if __name__ == "__main__":
    
    # Setup data generator class and load it into a pytorch dataloader
    dataset = TimeSeriesDataset(data_dir="data/processed/train/", 
                                normalize=True, 
                                window_size=WINDOW_SIZE, 
                                slide_length=TRAIN_SLIDE)
    train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

    use_cuda = torch.cuda.is_available()
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
    test_dataset = TimeSeriesDataset(data_dir="data/processed/test/", 
                                     normalize=True, 
                                     external_transform=dataset.get_transform(), # Use same transform as for training
                                     window_size=WINDOW_SIZE, 
                                     slide_length=TEST_SLIDE)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True, **kwargs)
    
    # Get input size and shapes
    input_size = dataset.get_input_layer_size_flattened()
    input_shape = dataset.get_input_layer_shape()
    channels = input_shape[0]
    height = input_shape[1]
    width = input_shape[2]
    
    logging.debug(f"input_shape: {input_shape}")
    logging.debug(f'channels: {channels}, height: {height}, width: {width}')

    # TODO: Do it a bit more smart, for now the height means the number of nodes in data
    ENCODER_LAYERS = ENCODER_LAYERS * height
    DECODER_LAYERS = DECODER_LAYERS * height 
    
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
    })

    # BUILD ENCODER
    encoder = nn.Sequential()
    encoder.append(nn.Linear(channels * width * height, ENCODER_LAYERS[0]))
    encoder.append(nn.GELU())
    
    for i in range(1, len(ENCODER_LAYERS)):
        encoder.append(nn.Linear(ENCODER_LAYERS[i-1], ENCODER_LAYERS[i]))
        encoder.append(nn.GELU())
        
    encoder.append(nn.Linear(ENCODER_LAYERS[-1], LATENT_DIM))
    logging.debug(f'Encoder Summary: {encoder}')
    
    # BUILD DECODER
    decoder = nn.Sequential()
    decoder.append(nn.Linear(LATENT_DIM, DECODER_LAYERS[0]))
    decoder.append(nn.GELU())

    for i in range(1, len(DECODER_LAYERS)):
        decoder.append(nn.Linear(DECODER_LAYERS[i-1], DECODER_LAYERS[i]))
        decoder.append(nn.GELU())
        
    decoder.append(nn.Linear(DECODER_LAYERS[-1], channels * width * height))
    logging.debug(f'Decoder Summary: {decoder}')

    # Init the lightning autoencoder
    autoencoder = LitAutoEncoder(input_shape, LATENT_DIM, encoder, decoder, lr=1e-4)
    
    # Add early stopping
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", 
        min_delta=0.00, 
        patience=10, 
        verbose=False, 
        mode="min"
        )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=1, verbose=True, monitor="val_loss", mode="min")


    logging.debug(f'Autoencoder Summary: {autoencoder}')
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        logger=logger,
        callbacks=[
            early_stop_callback,
            checkpoint_callback
            ],   
        enable_checkpointing=True,
        # accelerator="gpu", devices=1, strategy="auto",
        )
    trainer.fit(
        model=autoencoder, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
        )

    logging.debug(os.path.join(logger.save_dir, logger.name, logger.version, "checkpoints"))
    
    autoencoder = LitAutoEncoder.load_from_checkpoint(
                                checkpoint_path=checkpoint_callback.best_model_path,
                                encoder=encoder, 
                                decoder=decoder,
                                input_shape=input_shape,
                                latent_dim=LATENT_DIM
                                )
    
    # ----- TEST ----- #
    
    # Now test the model on february data
    # Run the model on the entire test set and report reconstruction error to tensorboard
    autoencoder.eval()
    autoencoder.freeze()
    
    logging.debug(f"Running model on test set")

    test_reconstruction_mean_absolute_error = []
    # Run evaluation on test set
    for idx, batch in tqdm.tqdm(enumerate(test_dataloader), desc="Running test reconstruction error", total=len(test_dataloader)):
        print(batch)
        err = torch.mean(torch.abs(batch - autoencoder.decoder(autoencoder.encoder(batch)))).detach().numpy()
        logger.experiment.add_scalar("test_reconstruction_error", err, idx)
        test_reconstruction_mean_absolute_error.append(err)
    

    # Plot reconstruction error over time
    dates_range = test_dataset.get_dates_range()
    logging.debug(f'Date range: {dates_range["end"]} - {dates_range["start"]}')
    dates_range = pd.date_range(start=dates_range["start"], end=dates_range["end"], freq='1min', tz='Europe/Warsaw')
    dates_range = dates_range.to_numpy()[:len(test_reconstruction_mean_absolute_error)] # Fit dates range to actual data (bear in mind that last date is max - WINDOW_SIZE)
    logging.debug(f"Plotting reconstruction error over time")
    plot_reconstruction_error_over_time(
        reconstruction_errors=test_reconstruction_mean_absolute_error,
        time_axis=dates_range,
        write=True,
        show=False,
        savedir=image_save_path
    )

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
    logging.debug(f"Plotting reconstructions vs real")
    sample = test_dataset.get_time_series()
    sample = torch.Tensor(sample[:, :, 0:WINDOW_SIZE].flatten())
    reconstructions = autoencoder.decoder(autoencoder.encoder(sample)).detach().numpy()
    sample = sample.detach().numpy()
    plot_embeddings_vs_real(
        _embeddings=reconstructions,
        _real=sample,
        channels=channels,
        height=height,
        width=width,
        checkpoint=checkpoint_callback.best_model_path,
        image_save_path=image_save_path,
        write=True,
        show=False,
    )


    logging.debug(f"Finished")
    
