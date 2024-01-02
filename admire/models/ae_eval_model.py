# -- Python imports --
import logging
import os
import yaml
import ast
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from math import ceil
from scipy.stats import zscore

# -- Pytorch imports --
import torch
from torch.utils.data import DataLoader
from lightning.pytorch.accelerators import CPUAccelerator
import lightning as L

# -- Project imports --
from ae_encoder import CNN_encoder, CNN_LSTM_encoder
from ae_decoder import CNN_decoder, CNN_LSTM_decoder
from ae_litmodel import LitAutoEncoder
from ae_dataloader import TimeSeriesDataset
from utils.plotting import *
from utils.config_reader import read_config


# ------------------ #


def setup_dataloader() -> tuple[DataLoader, TimeSeriesDataset]:
    train_dataset = TimeSeriesDataset(data_dir=f"{PROCESSED_PATH}/train/", 
                                    normalize=True, 
                                    window_size=WINDOW_SIZE, 
                                    slide_length=TRAIN_SLIDE)

    test_dataset = TimeSeriesDataset(data_dir=f"{PROCESSED_PATH}/test/", 
                                        normalize=True, 
                                        external_transform=train_dataset.get_transform(), # Use same transform as for training
                                        window_size=WINDOW_SIZE, 
                                        slide_length=TEST_SLIDE)
    
    
   
    kwargs = {'num_workers': 1, 'pin_memory': True} if USE_CUDA else {'num_workers': CPUAccelerator().auto_device_count(), 'pin_memory': False}
        
    test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=False, **kwargs)
    
    return test_dataloader, test_dataset

def setup_model(input_shape: tuple) -> tuple[LitAutoEncoder]:
    
    match MODEL_TYPE:
        case 'CNN':
            cnn_encoder = CNN_encoder(kernel_size=3, latent_dim=LATENT_DIM, cpu_alloc=INCLUDE_CPU_ALLOC, channels=ENCODER_LAYERS)
            cnn_decoder = CNN_decoder(kernel_size=3, latent_dim=LATENT_DIM, cpu_alloc=INCLUDE_CPU_ALLOC, channels=DECODER_LAYERS)

            autoencoder = LitAutoEncoder.load_from_checkpoint(
                                        checkpoint, 
                                        encoder=cnn_encoder, 
                                        decoder=cnn_decoder,
                                        input_shape=input_shape,
                                        latent_dim=LATENT_DIM,
                                        map_location=DEVICE
                                        )
        
        case 'LSTM':
            SEQUENCE_LENGTH = int(model_parameters['sequence_length'])
            LSTM_OUT_DIM = int(model_parameters['lstm_out_dim'])
            LSTM_IN_DIM = int(model_parameters['lstm_in_dim'])
            LSTM_HIDDEN_CHANNELS = list(ast.literal_eval(model_parameters['lstm_hidden_channels']))
            cnn_lstm_encoder = CNN_LSTM_encoder(lstm_input_dim=1, lstm_out_dim=LSTM_OUT_DIM, h_lstm_chan=LSTM_HIDDEN_CHANNELS, cpu_alloc=INCLUDE_CPU_ALLOC)
            cnn_lstm_decoder = CNN_LSTM_decoder(lstm_input_dim=LSTM_IN_DIM, lstm_out_dim=1, h_lstm_chan=LSTM_HIDDEN_CHANNELS, cpu_alloc=INCLUDE_CPU_ALLOC, seq_len=SEQUENCE_LENGTH)
            autoencoder = LitAutoEncoder.load_from_checkpoint(
                                        checkpoint, 
                                        encoder=cnn_lstm_encoder, 
                                        decoder=cnn_lstm_decoder,
                                        input_shape=input_shape,
                                        latent_dim=LATENT_DIM,
                                        map_location=DEVICE
                                        )

    return autoencoder

def run_test(autoencoder: L.LightningModule, 
             test_dataloader: DataLoader, 
             test_dataset: TimeSeriesDataset,
             test_date_range: np.ndarray,
             nodes_count: int,
             save_rec_err_to_parquet: bool = True,
             plot_rec_err: bool = True,
             test_batch_size: int = 1,
             device: str = 'cpu',
             save_eval_path: str = 'eval'
             ) -> np.ndarray:
    logging.debug(f"Running model on test set")

    test_recon_mae = []
    node_len = test_dataset.get_node_len()
    full_node_len = test_dataset.get_node_full_len()

    logging.debug(f"Node len: {node_len}, Full node len: {full_node_len}")

    # Run evaluation on test set
    for idx, batch in tqdm.tqdm(enumerate(test_dataloader), desc="Running test reconstruction error", total=ceil(len(test_dataset) / test_batch_size)):
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
    for i in range(nodes_count):
        test_recon_mae_stripped.append(test_recon_mae[(i*full_node_len): (node_len*(i+1)+(full_node_len-node_len)*i)])
        
    logging.debug(f"Test reconstruction error stripped len: {len(test_recon_mae_stripped)}")

    test_recon_mae_np = np.reshape(test_recon_mae_stripped, (nodes_count, node_len))
    test_recon_mae_list = test_recon_mae_np.tolist()
    agg_recon_err = np.mean(test_recon_mae_np, axis=0)

    hostnames = test_dataset.get_filenames()

    if save_rec_err_to_parquet:
        # Save reconstruction error to parquet
        try:
            stats_df = pd.DataFrame(test_recon_mae_np, index=hostnames, columns=test_date_range[0:len(test_recon_mae_np[0])].astype(str))
            stats_df.to_parquet(os.path.join(save_eval_path, 'recon_error.parquet'))
        except:
            logging.error('Error while saving recon_error to parquet')
            
    if plot_rec_err:
        # Display the reconstruction error over time manually
        logging.debug(f"Plotting reconstruction error over time")

        plot_recon_error_each_node(reconstruction_errors = test_recon_mae_list, 
                                time_axis = test_date_range, 
                                n_nodes = nodes_count, 
                                hostnames = hostnames, 
                                savedir=save_eval_path
                                )

        plot_recon_error_agg(reconstruction_errors = agg_recon_err, 
                            time_axis = test_date_range, 
                            hostnames = 'mean recon_error', 
                            savedir=save_eval_path
                            )        
    
    return test_recon_mae_np

def plot_z_scores(test_recon_mae_np: np.ndarray,
                test_dataset: TimeSeriesDataset,
                test_date_range: np.ndarray,
                save_eval_path: str
    ) -> None:
    # ----- Z-SCORES ----- #
    node_len = test_dataset.get_node_len()
    hostnames = test_dataset.get_filenames()
    
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

    

if __name__ == "__main__":
    
    USE_CUDA = torch.cuda.is_available()
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
    test_dataloader, test_dataset = setup_dataloader()
    
    test_len = len(test_dataset)
    d = next(iter(test_dataloader))
    input_shape = d.shape
    
    autoencoder = setup_model(input_shape=input_shape)
    autoencoder.eval()
    autoencoder.freeze()

    test_date_range = test_dataset.get_dates_range()
    test_date_range = pd.date_range(start=test_date_range['start'], end=test_date_range['end'], freq=f'{TEST_SLIDE}min', tz='Europe/Warsaw') # TODO set frequency dynamically?

    test_recon_mae_np = run_test(autoencoder=autoencoder,
                                test_dataloader=test_dataloader,
                                test_dataset=test_dataset,
                                test_date_range=test_date_range,
                                nodes_count=NODES_COUNT,
                                save_rec_err_to_parquet=True,
                                plot_rec_err=PLOT_REC_ERROR
                                )

    logging.debug(f'Dates range test: start {test_date_range.min()} end {test_date_range.max()}')
    logging.debug(f'Test len: {len(test_dataloader)}, Dates len: {len(test_date_range)}, Recon len: {test_recon_mae_np.shape}')

    if PLOT_ZSCORES:
        plot_z_scores(test_recon_mae_np=test_recon_mae_np,
                    test_dataset=test_dataset,
                    test_date_range=test_date_range,
                    save_eval_path=save_eval_path
                    )
