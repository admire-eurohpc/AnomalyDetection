# -- Python imports --
import logging
import os
import yaml
import ast
import pandas as pd
import numpy as np
import tqdm
import gc
import sys
import matplotlib.pyplot as plt
from math import ceil
from scipy.stats import zscore
import torch.multiprocessing as mp
from itertools import chain
import time


# -- Pytorch imports --
import torch
from torch.utils.data import DataLoader
from lightning.pytorch.accelerators import CPUAccelerator
from torch.distributions import Categorical


# -- Project imports --
from ae_encoder import CNN_encoder, CNN_LSTM_encoder
from ae_decoder import CNN_decoder, CNN_LSTM_decoder
from ae_litmodel import LitAutoEncoder
from ae_dataloader import TimeSeriesDataset
from data_exploration import kmeans_clustering
from utils.plotting import subplots_clustering
from utils.plotting import *
from utils.config_reader import read_config

gc.collect()
torch.cuda.empty_cache()
torch.multiprocessing.set_sharing_strategy('file_system')


USE_CUDA = torch.cuda.is_available()
print('using gpu: ', USE_CUDA)
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')

# -- Import settings and setup --
config_dict = read_config()
TEST_BATCH_SIZE = int(config_dict['EVALUATION']['BATCH_SIZE'])
LOGS_PATH = config_dict['EVALUATION']['logs_path']
PLOT_ZSCORES = config_dict['EVALUATION']['plot_zscores'].lower() == 'true'
PLOT_REC_ERROR = config_dict['EVALUATION']['plot_rec_error'].lower() == 'true'
# ------------------ #

# -- Load parameters from hparams.yaml --
with open(os.path.join(os.getcwd(), LOGS_PATH, 'hparams.yaml'), 'r') as f:
    params = yaml.load(f, yaml.UnsafeLoader)
    
model_key = list(filter(lambda x: 'PARAMETERS' in x, params.keys()))[0]
if model_key == None:
    raise Exception('No model parameters found in hparams.yaml')

model_parameters = params[model_key]
LATENT_DIM = int(model_parameters['latent_dim'])
TEST_SLIDE = int(model_parameters['test_slide'])
WINDOW_SIZE = int(model_parameters['window_size'])
TRAIN_SLIDE = int(model_parameters['train_slide'])
ENCODER_LAYERS = ast.literal_eval(model_parameters['encoder_layers'])
DECODER_LAYERS = ast.literal_eval(model_parameters['decoder_layers'])

NODES_COUNT = int(params['PREPROCESSING']['nodes_count_to_process'])
INCLUDE_CPU_ALLOC = params['PREPROCESSING']['with_cpu_alloc'].lower() == 'true'
PROCESSED_PATH = params['PREPROCESSING']['processed_data_dir']
TEST_DATES_RANGE = params['PREPROCESSING']['test_date_range']

MODEL_TYPE = params['TRAINING']['model_type']

path = os.path.join(os.getcwd(), LOGS_PATH, 'checkpoints')
save_eval_path = os.path.join(os.getcwd(), LOGS_PATH, f'eval-{TEST_DATES_RANGE.split(",")}')

if not os.path.exists(save_eval_path):
    os.makedirs(save_eval_path)

filenames = os.walk(path).__next__()[2] # get any checkpoint
last_epoch_idx = np.array([int(i.split('-')[0].split('=')[1]) for i in filenames]).argmax()
filename = filenames[last_epoch_idx]

checkpoint = os.path.join(path, filename)

logging.basicConfig(level=logging.DEBUG)

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

# ------------------ #

ENTROPY_WINDOW_SIZE = 180

def multiprocess_batch(batch: torch.Tensor):
    entropy = []
    for i in range(0,batch.shape[0]):
        _,counts=torch.unique(batch[i].data.flatten(start_dim=0),dim=0,return_counts=True)
        p=(counts)/float(batch.shape[1]*batch.shape[2])
        entropy.append(float(torch.sum(p* torch.log2(p))*-1))
    return entropy




def setup_dataloader() -> tuple[DataLoader, TimeSeriesDataset]:
    # train_dataset = TimeSeriesDataset(data_dir=f"{PROCESSED_PATH}/train/", 
    #                                 normalize=True, 
    #                                 window_size=ENTROPY_WINDOW_SIZE, 
    #                                 slide_length=ENTROPY_WINDOW_SIZE)

    test_dataset = TimeSeriesDataset(data_dir=f"{PROCESSED_PATH}/test/", 
                                        normalize=True, 
                                        #external_transform=train_dataset.get_transform(), # Use same transform as for training
                                        window_size=WINDOW_SIZE, 
                                        slide_length=10)
    
    
   
    kwargs = {'num_workers': 1, 'pin_memory': True} if USE_CUDA else {'num_workers': CPUAccelerator().auto_device_count(), 'pin_memory': False}
        
    test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=False, **kwargs)
    
    return test_dataset, test_dataloader

if __name__ == "__main__":
    test_dataset, test_dataloader = setup_dataloader()

    print(len(test_dataset))
    entropy_list = []
    node_len = test_dataset.get_node_len()
    full_node_len = test_dataset.get_node_full_len()
    print("node len: ", node_len, "full_node_len: ", full_node_len)

    processes = []
    with mp.Pool(24) as pool:
        for x in tqdm.tqdm(pool.imap(multiprocess_batch, test_dataloader),
                                desc="Running entropy calculation reconstruction error", 
                                total=len(test_dataloader)):
            entropy_list.append(x)


    entropy_list= list(chain.from_iterable(entropy_list))

    entropy_stripped = []
    for i in range(NODES_COUNT):
        print(len(entropy_list[(i*full_node_len): (i*full_node_len) + node_len]))
        entropy_stripped.append(entropy_list[(i*full_node_len): (i*full_node_len) + node_len])
        
    logging.debug(f"Test reconstruction error stripped len: {len(entropy_stripped)}")

    test_recon_mae_np = np.reshape(entropy_stripped, (NODES_COUNT, node_len))
    test_recon_mae_list = test_recon_mae_np.tolist()

    plot_recon_error_each_node(reconstruction_errors = test_recon_mae_list, 
                                time_axis = test_dataset.get_dates_range(), 
                                n_nodes = 200, 
                                hostnames = test_dataset.get_filenames(), 
                                savedir=save_eval_path
                                )

