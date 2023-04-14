import logging
import os
import yaml
from torch import nn
import torch
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import configparser

from ae_litmodel import LitAutoEncoder
from ae_dataloader import TimeSeriesDataset
from utils.plotting import *
LOGS_PATH = 'lightning_logs/ae/2023_04_14-21_25_33'
path = os.path.join(os.getcwd(), LOGS_PATH, 'checkpoints')

config = configparser.ConfigParser()
config.read('config.ini')

print(os.path.exists(path))

filenames = os.walk(path).__next__()[2] # get any checkpoint
last_epoch_idx = np.array([int(i.split('-')[0].split('=')[1]) for i in filenames]).argmax()
filename = filenames[last_epoch_idx]

print(filename)


checkpoint = os.path.join(path, filename)

with open(os.path.join(os.getcwd(), LOGS_PATH, 'hparams.yaml'), 'r') as f:
    params = yaml.load(f, yaml.UnsafeLoader)

logging.basicConfig(level=logging.DEBUG)

print(params)
channels = params['number_of_channels']
height = params['number_of_nodes']
width = params['window_size']
input_shape = (channels, height, width)

ENCODER_LAYERS = params['encoder_layers']
DECODER_LAYERS = params['decoder_layers']
LATENT_DIM = params['latent_dim']
TEST_SLIDE = params['test_slide']
WINDOW_SIZE = params['window_size']
TRAIN_SLIDE = params['train_slide']

PROCESSED_PATH = config.get('PREPROCESSING', 'processed_data_dir')


use_cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

train_dataset = TimeSeriesDataset(data_dir=f"{PROCESSED_PATH}/train/", 
                                normalize=True, 
                                window_size=WINDOW_SIZE, 
                                slide_length=TRAIN_SLIDE)
# Setup test data
test_dataset = TimeSeriesDataset(data_dir=f"{PROCESSED_PATH}/test/", 
                                    normalize=True, 
                                    external_transform=train_dataset.get_transform(), # Use same transform as for training
                                    window_size=WINDOW_SIZE, 
                                    slide_length=TEST_SLIDE)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True, **kwargs)

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


autoencoder = LitAutoEncoder.load_from_checkpoint(
                            checkpoint, 
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

test_date_range = test_dataset.get_dates_range()
test_date_range = pd.date_range(start=test_date_range['start'], end=test_date_range['end'], freq=f'{TEST_SLIDE}min', tz='UTC') # TODO set frequency dynamically?

logging.debug(f'Dates range test: start {test_date_range.min()} end {test_date_range.max()}')
logging.debug(f'Test len: {len(test_dataloader)}, Dates len: {len(test_date_range)}')

test_reconstruction_mean_absolute_error = []
for idx, batch in tqdm.tqdm(enumerate(test_dataloader), desc="Running test reconstruction error evaluation", total=len(test_dataloader)):
    err = torch.sum(torch.abs(batch - autoencoder.decoder(autoencoder.encoder(batch)))).detach().numpy()
    test_reconstruction_mean_absolute_error.append(err)

# Display the reconstruction error over time manually
logging.debug(f'Test reconstruction instances: {len(test_reconstruction_mean_absolute_error)}')

logging.debug(f"Plotting reconstruction error over time")

figure, ax = plt.subplots(1,1, figsize=(10, 8))
ax.scatter(test_date_range.to_numpy()[:len(test_reconstruction_mean_absolute_error)], test_reconstruction_mean_absolute_error)
ax.set_xlabel('Date')
ax.set_ylabel('Reconstruction error')
ax.set_title('Reconstruction error (test data)\nDate on x-axis corresponds to the *start* of the window')
plt.xticks(rotation=70)
plt.tight_layout()

savedir = os.path.join(os.getcwd(), 'images', 'reconstruction')
if not os.path.exists(savedir):
    os.makedirs(savedir)

plt.savefig(os.path.join(savedir, 'plt_reconstruction.png'))

print('Plotly')

plot_reconstruction_error_over_time(test_reconstruction_mean_absolute_error, 
                                    test_date_range.to_numpy()[:len(test_reconstruction_mean_absolute_error)], 
                                    show=True, 
                                    write=True,
                                    savedir=savedir)