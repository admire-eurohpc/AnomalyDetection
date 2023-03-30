import logging
import os 
import pandas as pd
from datetime import datetime
import numpy as np

# -- Pytorch imports --
import torch
from torch.utils.data import DataLoader
from torch import optim, nn, utils, Tensor
from torchsummary import summary
import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger

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


BATCH_SIZE = 64
MAX_EPOCHS = 100
SHUFFLE = True


# THIS NUMBERS WILL BE MULTIPLIED BY NUMBER OF NODES 
ENCODER_LAYERS = np.array([32, 16])
DECODER_LAYERS = np.array([16, 32])
LATENT_DIM = 8 


def calculate_linear_layer_size_after_conv(input_shape, kernel_size, stride, padding):
    '''
    Calculates the size of the linear layer after a convolutional layer.
    
    input_shape: tuple of ints (channels, height, width)
    '''
    channels, height, width = input_shape
    height = (height - kernel_size + 2*padding) / stride + 1
    width = (width - kernel_size + 2*padding) / stride + 1
    return int(channels * height * width)



if __name__ == "__main__":
    
    # Setup data generator class and load it into a pytorch dataloader
    dataset = TimeSeriesDataset(data_dir="data/processed/train/", normalize=True)
    train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE
        )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE
        )
    
    test_dataset = TimeSeriesDataset(data_dir="data/processed/test/", normalize=True, external_transform=dataset.get_transform())
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    input_size = dataset.get_input_layer_size_flattened()

    input_shape = dataset.get_input_layer_shape()
    channels = input_shape[0]
    height = input_shape[1]
    width = input_shape[2]
    
    logging.debug(f"input_shape: {input_shape}")
    logging.debug(f'channels: {channels}, height: {height}, width: {width}')

    ENCODER_LAYERS = ENCODER_LAYERS * height
    DECODER_LAYERS = DECODER_LAYERS * height # TODO: Do it a bit more smart, for now the height means the number of nodes in data

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

    # Init the autoencoder
    autoencoder = LitAutoEncoder(input_shape, LATENT_DIM, encoder, decoder)
    
    # Add early stopping
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", 
        min_delta=0.00, 
        patience=5, 
        verbose=False, 
        mode="min"
        )

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        logger=logger,
        callbacks=[
            early_stop_callback,
            ],
        )
    trainer.fit(
        model=autoencoder, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
        )

    print(os.path.join(logger.save_dir, logger.name, logger.version, "checkpoints"))
    # load checkpoint
    path = os.path.join(logger.save_dir, logger.name, logger.version, "checkpoints")
    filename = os.walk(path).__next__()[2][0]
    checkpoint = os.path.join(path, filename)
    

    autoencoder = LitAutoEncoder.load_from_checkpoint(
                                checkpoint, 
                                encoder=encoder, 
                                decoder=decoder,
                                input_shape=input_shape,
                                latent_dim=LATENT_DIM
                                )

    


    # ----- TEST ----- #
    
    # Now test the model on february data
    # Run the model on the entire test set and report MAE 
    encoder = autoencoder.encoder.eval()
    decoder = autoencoder.decoder.eval()

    test_reconstruction_mean_absolute_error = []
    for batch in test_dataloader:
        test_reconstruction_mean_absolute_error.append(
            torch.sum(torch.abs(batch - decoder(encoder(batch)))).detach().numpy()
        )
    
    dates = pd.read_parquet("data/processed/test/test_e1105.parquet")['date']
    dates = pd.to_datetime(dates)
    logging.debug(f'Date range: {dates.min()} - {dates.max()}')
    
    dates = [i for i in range(len(test_reconstruction_mean_absolute_error))]
    
    logging.debug(f"Plotting reconstruction error over time")
    plot_reconstruction_error_over_time(
        reconstruction_errors=test_reconstruction_mean_absolute_error,
        time_axis=dates
    )


    some_sample = next(iter(test_dataloader))
    reconstructions = decoder(encoder(some_sample))
    logging.debug(f"Plotting reconstructions vs real")
            
    plot_embeddings_vs_real(
        _embeddings=reconstructions.detach().numpy(),
        _real=some_sample.detach().numpy(),
        channels=channels,
        height=height,
        width=width,
        checkpoint=checkpoint,
        image_save_path=image_save_path,
    )
    
