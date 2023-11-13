import logging
import os
import yaml
import torch
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from math import ceil
from torch.utils.data import DataLoader
from scipy.stats import zscore
from lightning.pytorch.accelerators import CPUAccelerator
import configparser

from ae_encoder import CNN_encoder, CNN_LSTM_encoder
from ae_decoder import CNN_decoder, CNN_LSTM_decoder
from ae_litmodel import LitAutoEncoder
from ae_dataloader import TimeSeriesDataset
from utils.plotting import *

config = configparser.ConfigParser()
config.read('config.ini')

PROCESSED_PATH = config.get('PREPROCESSING', 'processed_data_dir')
INCLUDE_CPU_ALLOC = config.getboolean('PREPROCESSING', 'with_cpu_alloc')
LOGS_PATH = config.get('EVALUATION', 'logs_path')
NODES_COUNT = config.getint('PREPROCESSING', 'nodes_count_to_process')
TEST_DATES_RANGE = config.get('PREPROCESSING', 'test_date_range')

path = os.path.join(os.getcwd(), LOGS_PATH, 'checkpoints')
save_eval_path = os.path.join(os.getcwd(), LOGS_PATH, f'eval-{TEST_DATES_RANGE.split(",")}')

if not os.path.exists(save_eval_path):
    os.makedirs(save_eval_path)

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
ENCODER_LAYERS = eval(params['encoder_layers'])
DECODER_LAYERS = eval(params['decoder_layers'])

TEST_BATCH_SIZE = 4800

#cuda not required for inference on batch = 1
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {'num_workers': CPUAccelerator().auto_device_count(), 'pin_memory': False}

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
test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=False, **kwargs)

test_len = len(test_dataset)
d = next(iter(test_dataloader))
input_shape = d.shape

cnn_lstm_encoder = CNN_LSTM_encoder(lstm_input_dim=1, lstm_out_dim=48, h_lstm_chan=[96], cpu_alloc=True)
cnn_lstm_decoder = CNN_LSTM_decoder(lstm_input_dim=48, lstm_out_dim =1, h_lstm_chan=[96], cpu_alloc=True)
lstm_conv_autoencoder = LitAutoEncoder(cnn_lstm_encoder, cnn_lstm_decoder)

cnn_encoder = CNN_encoder(kernel_size=3, latent_dim=LATENT_DIM, cpu_alloc=INCLUDE_CPU_ALLOC, channels=ENCODER_LAYERS)
cnn_decoder = CNN_decoder(kernel_size=3, latent_dim=LATENT_DIM, cpu_alloc=INCLUDE_CPU_ALLOC, channels=DECODER_LAYERS)

autoencoder = LitAutoEncoder.load_from_checkpoint(
                            checkpoint, 
                            encoder=cnn_lstm_encoder, 
                            decoder=cnn_lstm_decoder,
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

logging.debug(f"Node len: {node_len}, Full node len: {full_node_len}")

# Run evaluation on test set
for idx, batch in tqdm.tqdm(enumerate(test_dataloader), desc="Running test reconstruction error", total=ceil(test_len / TEST_BATCH_SIZE)):
    batch = batch.to(device)
    batch_err = torch.abs(batch - autoencoder.decoder(autoencoder.encoder(batch)))
    
    err = torch.mean(batch_err, dim=(1,2))
    
    err_detached = err.cpu().numpy()
    
    # Flatten the error tensor
    err_detached = err_detached.flatten()
    
    
    test_recon_mae += err_detached.tolist()

logging.debug(f"Test reconstruction error len: {len(test_recon_mae)}")

'''Due to processing whole test set at once (one node after one, last WINDOW_SIZE-1 samples of node x
 are processed together with samples from x+1 node resulting in high reconstruction errors on the beginning and the end of test_set)
 On training set this is also present but considering vast amount of samples in each node the impact is insignificant'''

# ----- RECONSTRUCTION ERROR ----- #

test_recon_mae_stripped = []
for i in range(NODES_COUNT):
    test_recon_mae_stripped.append(test_recon_mae[(i*full_node_len): (node_len*(i+1)+(full_node_len-node_len)*i)])
    
logging.debug(f"Test reconstruction error stripped len: {len(test_recon_mae_stripped)}")

test_recon_mae_np = np.reshape(test_recon_mae_stripped, (NODES_COUNT, node_len))
test_recon_mae_list = test_recon_mae_np.tolist()
agg_recon_err = np.mean(test_recon_mae_np, axis=0)
agg_recon_err_2 = np.median(test_recon_mae_np, axis=0)
agg_recon_err_3 = np.quantile(test_recon_mae_np, 0.25, axis=0)
agg_recon_err_4 = np.quantile(test_recon_mae_np, 0.75, axis=0)


test_date_range = test_dataset.get_dates_range()
test_date_range = pd.date_range(start=test_date_range['start'], end=test_date_range['end'], freq=f'{TEST_SLIDE}min', tz='Europe/Warsaw') # TODO set frequency dynamically?

logging.debug(f'Dates range test: start {test_date_range.min()} end {test_date_range.max()}')
logging.debug(f'Test len: {len(test_dataloader)}, Dates len: {len(test_date_range)}, Recon len: {len(test_recon_mae)}')

hostnames = test_dataset.get_filenames()

# Save reconstruction error to parquet
try:
    print(test_recon_mae_np.shape)
    print(test_date_range.shape)
    print(len(hostnames))
    stats_df = pd.DataFrame(test_recon_mae_np, index=hostnames, columns=test_date_range[0:len(test_recon_mae_np[0])].astype(str))
    stats_df.to_parquet(os.path.join(save_eval_path, 'recon_error.parquet'))
except:
    print('Error while saving recon_error to parquet')

# Display the reconstruction error over time manually
logging.debug(f"Plotting reconstruction error over time")

plot_recon_error_each_node(reconstruction_errors = test_recon_mae_list, 
                           time_axis = test_date_range, 
                           n_nodes = 200, 
                           hostnames = hostnames, 
                           savedir=save_eval_path
                           )

plot_recon_error_agg(reconstruction_errors = agg_recon_err, 
                     time_axis = test_date_range, 
                     hostnames = 'mean recon_error', 
                     savedir=save_eval_path
                     )


# ----- Z-SCORES ----- #

#calculate threshold metric z-score for anomaly evaluation
zscores = np.zeros((NODES_COUNT, node_len))
for i in range(node_len):
    zscores[:, i] = (zscore(test_recon_mae_np[:, i]))
    
    
plot_recon_error_each_node(reconstruction_errors = zscores, 
                           time_axis = test_date_range, 
                           n_nodes = 200, 
                           hostnames = hostnames, 
                           savedir=save_eval_path, 
                           out_name = 'zscores_all'
                           )
plot_recon_error_agg(reconstruction_errors = np.mean(zscores, axis=0), 
                     time_axis = test_date_range, 
                     hostnames = 'mean zscore', 
                     savedir=save_eval_path
                     )    

# Wrapping this in try-except because it sometimes fails to save the parquet file and I don't know why
try: 
    print(zscores.shape)
    print(test_date_range.shape)
    print(len(hostnames))    
    stats_df = pd.DataFrame(zscores, index=hostnames, columns=test_date_range[0:len(zscores[0])].astype(str))
    stats_df.to_parquet(os.path.join(save_eval_path, 'zscores.parquet'))
except:
    print('Error while saving zscores to parquet')

# 3 standard deviations from mean
z_thresholds = [1, 2, 3]
for z_threshold in z_thresholds:
    zscores_thresholded = zscores.copy()
    zscores_thresholded[zscores < z_threshold] = 0
    zscores_thresholded[zscores >= z_threshold] = 1

    print(zscores_thresholded.shape)

    plot_recon_error_each_node(reconstruction_errors = zscores_thresholded, 
                            time_axis = test_date_range, 
                            n_nodes = 200, 
                            hostnames = hostnames, 
                            savedir=save_eval_path, 
                            out_name = f'zscores_thresholded_all (s={z_threshold})'
                            )

    # Display the sum of thresholded z-scores over time  
    logging.debug(f"Plotting thresholded z-scores over time")
    s = np.sum(zscores_thresholded, axis=0)
    plot_recon_error_agg(reconstruction_errors = s, 
                        time_axis = test_date_range, 
                        hostnames = f'sum of thresholded (s={z_threshold}) zscores', 
                        savedir=save_eval_path
                        )

