import logging
import os
import yaml
import torch
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.stats import zscore
import configparser

from ae_encoder import CNN_encoder
from ae_decoder import CNN_decoder
from ae_litmodel import LitAutoEncoder
from ae_dataloader import TimeSeriesDataset
from utils.plotting import *

config = configparser.ConfigParser()
config.read('config.ini')

PROCESSED_PATH = config.get('PREPROCESSING', 'processed_data_dir')
INCLUDE_CPU_ALLOC = config.getboolean('PREPROCESSING', 'with_cpu_alloc')
LOGS_PATH = config.get('EVALUATION', 'logs_path')
NODES_COUNT = config.getint('PREPROCESSING', 'nodes_count_to_process')

path = os.path.join(os.getcwd(), LOGS_PATH, 'checkpoints')
print(os.path.exists(path))

filenames = os.walk(path).__next__()[2] # get any checkpoint
last_epoch_idx = np.array([int(i.split('-')[0].split('=')[1]) for i in filenames]).argmax()
filename = filenames[last_epoch_idx]

checkpoint = os.path.join(path, filename)

with open(os.path.join(os.getcwd(), LOGS_PATH, 'hparams.yaml'), 'r') as f:
    params = yaml.load(f, yaml.UnsafeLoader)

logging.basicConfig(level=logging.DEBUG)

print(params)

LATENT_DIM = params['latent_dim']
TEST_SLIDE = params['test_slide']
WINDOW_SIZE = params['window_size']
TRAIN_SLIDE = params['train_slide']

#cuda not required for inference on batch = 1
#use_cuda = torch.cuda.is_available()
use_cuda=False
device = torch.device('cuda:0' if use_cuda else 'cpu')
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

test_len = len(test_dataset)
d = next(iter(test_dataloader))
input_shape = d.shape

cnn_encoder = CNN_encoder(kernel_size=10, latent_dim=LATENT_DIM, cpu_alloc=INCLUDE_CPU_ALLOC)
cnn_decoder = CNN_decoder(latent_dim=LATENT_DIM, cpu_alloc=INCLUDE_CPU_ALLOC)

autoencoder = LitAutoEncoder.load_from_checkpoint(
                            checkpoint, 
                            encoder=cnn_encoder, 
                            decoder=cnn_decoder,
                            input_shape=input_shape,
                            latent_dim=LATENT_DIM,
                            map_location=device
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

#calculate threshold metric z-score for anomaly evaluation
zscores = np.zeros((NODES_COUNT, node_len))
for i in range(node_len):
    zscores[:, i] = (zscore(test_recon_mae_np[:, i]))

plot_recon_error_each_node(reconstruction_errors = zscores, time_axis = test_date_range, n_nodes = 200, hostnames = hostnames, savedir=LOGS_PATH, out_name = 'zscores')
