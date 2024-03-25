from collections import defaultdict
import copy
import numpy as np
from ae_encoder import CNN_encoder, CNN_LSTM_encoder
from ae_decoder import CNN_decoder, CNN_LSTM_decoder
from ae_litmodel import LitAutoEncoder, LSTM_AE
from ae_lstm_vae import LSTMVAE, LSTMEncoder, LSTMDecoder
from timeseriesdatasetv2 import TimeSeriesDatasetv2

import lightning as L 
import os
import wandb
import logging
import yaml
import ast
import torch

class ModelLoaderService:
    '''
    ModelLoaderService is a class that provides a unified interface for loading models from different sources.
    '''
    model: L.LightningModule = None
    supported_model_types: list[str] = ['CNN', 'LSTMCNN', 'LSTMVAE', 'LSTMPLAIN']
    model_type: str = None
    is_model_loaded: bool = False
    hyperparameters: dict = None
    
    def __init__(self, model_type: str, print_debug: bool = True) -> None:
        '''
        Parameters:
            - model_type (str): Type of the model to be used. 
            - print_debug (bool): Whether to print debug information. Default: True
        '''
        assert model_type.upper() in self.supported_model_types, \
            f"Model type {model_type} not supported. Supported model types: {self.supported_model_types}"
            
        self.model_type = model_type.upper()
        self.print_debug = print_debug
    
    def load_model_from_local(self, model_path: str, hparams: dict | str, device: str = 'cpu'):
        '''
        Load model from local path.
        '''
        raise NotImplementedError("Loading model from local path not implemented yet.")
        
    def load_model_from_wandb(self, 
                              run_id: str, 
                              entity: str = "ignacysteam",
                              project: str = "lightning_logs",
                              model_tag: str = "v0",
                              device: str = 'cpu',
        ) -> L.LightningModule:
        '''
        Load model from wandb.
        
        In general you probably should only pass the run_id and the rest of the parameters should be left as default.  
        Unless some routing changes were made.  
        Also, you should be able to get these parameters from the wandb interface for the run you are interested in.
        
        Parameters:
            - run_id (str): ID of the run in wandb.
            - entity (str): Name of the entity in wandb. Default: "ignacysteam"
            - project (str): Name of the project in wandb. Default: "lightning_logs"
            - model_tag (str): Tag of the model artifact. Default: "v0"
            - device (str): Device to load the model on. Default: 'cpu'
            
        Returns:
            - L.LightningModule: Model loaded from wandb. This model is also stored in the class attribute `model`.  
            
            
        Example:
            ```python
            modell_service = ModelLoaderService('LSTMCNN')
            modell_service.load_model_from_wandb(run_id='e_standard-2024_02_20-14_08_29')
            modell_service.print_model_summary()
            ```
        
        '''
        
        # Set up the API
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        run_config = run.config
        
        # Get the artifact and download the model to local directory
        artifact = api.artifact(f'{entity}/{project}/model-{run_id}:{model_tag}', type='model')
        path = artifact.download()
        path = os.path.join(path, 'model.ckpt')
        
        # Get the hyperparameters
        model_parameters = run_config['model_parameters'.upper()]
        self.hyperparameters = model_parameters
        
        # Load the model
        match self.model_type:
            case 'CNN':
                self.is_model_loaded = self.__load_CNN_model(path, model_parameters, device)
            case 'LSTMCNN':
                self.is_model_loaded = self.__load_LSTM_CNN_model(path, model_parameters, device)
            case 'LSTMVAE':
                self.is_model_loaded = self.__load_LSTM_VAE_model(path, model_parameters, device)
            case 'LSTMPLAIN':
                self.is_model_loaded = self.__load_LSTM_plain_model(path, model_parameters, device)
            case _:
                self.is_model_loaded = False
                raise ValueError(f"Model type {self.model_type} not supported.")
            
        return self.model
              
    def __load_LSTM_CNN_model(self, model_path: str, model_hparameters: dict, device: str = 'cpu') -> bool:
        '''
        Load the LSTM-CNN model from the checkpoint.
        
        Parameters:
            - model_path (str): Path to the model checkpoint.
            - model_hparameters (dict): Hyperparameters of the model.
            - device (str): Device to load the model on. Default: 'cpu'
            
        Returns:    
            - bool: True if the model was loaded successfully.
        '''
        # Extract parameters for the model
        lstm_hidden_channels = list(ast.literal_eval(model_hparameters['lstm_hidden_channels']))
        lstm_in_dim = int(model_hparameters['lstm_in_dim'])
        lstm_out_dim = int(model_hparameters['lstm_out_dim'])
        sequence_length = int(model_hparameters['sequence_length'])
        
        # Initialize the sub-models
        cnn_lstm_encoder = CNN_LSTM_encoder(lstm_input_dim=1, lstm_out_dim=lstm_out_dim, h_lstm_chan=lstm_hidden_channels)
        cnn_lstm_decoder = CNN_LSTM_decoder(lstm_input_dim=lstm_in_dim, lstm_out_dim=1, h_lstm_chan=lstm_hidden_channels, seq_len=sequence_length)
          
        # Load the model from the checkpoint
        self.model = LitAutoEncoder.load_from_checkpoint(model_path, map_location=device, encoder=cnn_lstm_encoder, decoder=cnn_lstm_decoder)
        if self.print_debug:
            print(self.model)
        
        # Load the weights of the sub-models
        # This is according to: https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html
        # Only necessary when the sub-models are passed as arguments to the main model
        # Otherwise, the weights are loaded automatically
        checkpoint = torch.load(model_path, map_location='cpu')
        
        encoder_weights = {self.__strip_key(k): v for k, v in checkpoint["state_dict"].items() if k.startswith("encoder.")}
        decoder_weights = {self.__strip_key(k): v for k, v in checkpoint["state_dict"].items() if k.startswith("decoder.")}
        
        # Load the weights
        self.model.encoder.load_state_dict(encoder_weights)
        self.model.decoder.load_state_dict(decoder_weights)
        
        return True
    
    def __load_LSTM_VAE_model(self, model_path: str, model_hparameters: dict, device: str = 'cpu') -> bool:
        raise NotImplementedError("LSTM VAE model not implemented yet.")
    
    def __load_LSTM_plain_model(self, model_path: str, model_hparameters: dict, device: str = 'cpu') -> bool:
        raise NotImplementedError("LSTM plain model not implemented yet.")
    
    def __load_CNN_model(self, model_path: str, model_hparameters: dict, device: str = 'cpu') -> bool:
        '''
        Load the CNN model from the checkpoint.
        
        Parameters:
            - model_path (str): Path to the model checkpoint.
            - model_hparameters (dict): Hyperparameters of the model.
            - device (str): Device to load the model on. Default: 'cpu'
            
        Returns:
            - bool: True if the model was loaded successfully.
        '''
        latent_dim = int(model_hparameters['latent_dim'])
        encoder_layers = list(ast.literal_eval(model_hparameters['encoder_layers']))
        decoder_layers = list(ast.literal_eval(model_hparameters['decoder_layers']))
        
        cnn_encoder = CNN_encoder(kernel_size=3, latent_dim=latent_dim, channels=encoder_layers)
        cnn_decoder = CNN_decoder(kernel_size=3, latent_dim=latent_dim, channels=decoder_layers)
        
        # Load the model from the checkpoint
        self.model = LitAutoEncoder.load_from_checkpoint(model_path, map_location=device, encoder=cnn_encoder, decoder=cnn_decoder)
        
        checkpoint = torch.load(model_path, map_location=device)
        encoder_weights = {k: v for k, v in checkpoint["state_dict"].items() if k.startswith("encoder.")}
        decoder_weights = {k: v for k, v in checkpoint["state_dict"].items() if k.startswith("decoder.")}
        
        # Load the weights
        self.model.encoder.load_state_dict(encoder_weights)
        self.model.decoder.load_state_dict(decoder_weights)
        
        return True

    @staticmethod
    def __strip_key(key: str) -> str:
        '''Fix mapping of the keys in the checkpoint to the model's keys.'''
        if key.startswith("encoder."):
            return key[8:]
        if key.startswith("decoder."):
            return key[8:]
        return key  
    
    def get_model(self) -> L.LightningModule:
        '''
        Get the model.
        
        Returns:
            - L.LightningModule: Model loaded from the source.
        '''
        return self.model
    
    def check_is_model_loaded(self) -> bool:
        '''
        Check if the model is loaded.
        
        Returns:
            - bool: True if the model is loaded.
        '''
        return self.is_model_loaded
    
    def get_model_type(self) -> str:
        '''
        Get the type of the model.
        
        Returns:
            - str: Type of the model.
        '''
        return self.model_type 
        
    def print_model_summary(self):
        '''
        Print the summary of the model.
        '''
        print(self.model)
  
  
class ModelInference:
    
    model: L.LightningModule = None
    model_type: str = None
    hyperparameters: dict = None
    device: str = None
     
    def __init__(self, 
        model: L.LightningModule,
        model_type: str,
        hyperparameters: dict,
        device: str = 'cpu'
        ) -> None:
        '''
        Parameters:
            - model (L.LightningModule): Model to be used for inference.
            - model_type (str): Type of the model.
            - hyperparameters (dict): Hyperparameters of the model.
            - device (str): Device to load the model on. Default: 'cpu'
        '''
        self.device = device
        self.hyperparameters = hyperparameters
        self.model = model
        self.model_type = model_type
        
        # Set the model to the device
        self.model.to(self.device)
        
        # Set the model to evaluation mode
        self.model.eval()
               
    def infer(self, data: torch.Tensor | np.ndarray) -> np.ndarray:
        '''
        Infer the data using the model.
        
        Parameters:
            - data (torch.Tensor | np.ndarray): Data to be inferred.
            
        Returns:
            - np.ndarray: Inferred data.
        '''
        # Convert the data to tensor if it is numpy array
        if isinstance(data, np.ndarray):
            data = self.__array_to_tensor(data)
            
        # Move the data to the device
        if data.device != self.device:
            data = data.to(self.device)
            
        # Infer the data
        with torch.no_grad():
            output: torch.Tensor = self.model(data)
            
        # Move the data and output back to the CPU
        # and numpy array 
        output = output.cpu().numpy()
        input_data = data.cpu().numpy()
        
        
        # Make sure that the output has the same shape as the input
        # Or try to reshape it
        if input_data.shape != output.shape:
            try:
                output = output.reshape(input_data.shape)
            except Exception as e:
                logging.error(f"Error during reshaping the output: {e}")
                raise ValueError("Output shape does not match the input shape.")
            
        return output
    
    def calculate_reconstruction_error(self, original_data: np.ndarray, reconstructed_data: np.ndarray, metric: str = 'L1', batch_axis: int = 0) -> dict:
        '''
        Compute the L1 reconstruction error between the original and reconstructed data.
        
        Parameters:
            - original_data (np.ndarray): Original data.
            - reconstructed_data (np.ndarray): Reconstructed data.
            - metric (str): Metric to be used for the reconstruction error. Default: 'L1'
            - batch_axis (int): Axis of the batch. Default: 0
            
        Returns:
            - dict
                - np.ndarray: Reconstruction error per_sample_and_channel. Shape is (n_samples, n_features).
                - np.ndarray: Reconstruction error per-sample. Shape is (n_samples,).
                - np.ndarray: Reconstruction error per-batch. Shape is (1,).
        '''
        assert original_data.shape == reconstructed_data.shape, \
            f"Original and reconstructed data must have the same shape. Original: {original_data.shape}, Reconstructed: {reconstructed_data.shape}"
            
        match metric:
            case 'L1':
                per_sample_and_channel = np.abs(original_data - reconstructed_data).mean(axis=2)
                error_per_sample = np.abs(original_data - reconstructed_data).mean(axis=(1, 2))
                error_per_batch = np.abs(original_data - reconstructed_data).mean()
            case 'L2':
                per_sample_and_channel = np.square(original_data - reconstructed_data).mean(axis=2)
                error_per_sample = np.square(original_data - reconstructed_data).mean(axis=(1, 2))
                error_per_batch = np.square(original_data - reconstructed_data).mean()
            case _:
                raise ValueError(f"Metric {metric} not supported.")
            
        return {
            'per_sample_and_channel': per_sample_and_channel,
            'per_sample': error_per_sample,
            'per_batch': error_per_batch
        }    
            
        
    def __array_to_tensor(self, data: np.ndarray) -> torch.Tensor:
        '''
        Convert numpy array to torch tensor.
        
        Parameters:
            - data (np.ndarray): Data to be converted.
            
        Returns:
            - torch.Tensor: Converted data.
        '''
        return torch.tensor(data, device=self.device)     
  
  
class DataLoaderService:
    
    dataset: TimeSeriesDatasetv2 = None
    dataset_length: int = None
    
    def __init__(self, 
        data_dir: str,
        data_normalization: bool = True,
        window_size: int = 60,
        slide_length: int = 1,
        nodes_count: int = 4
        ) -> None:
        
        self.dataset = TimeSeriesDatasetv2(
            data_dir=data_dir,
            normalize=data_normalization,
            window_size=window_size,
            slide_length=slide_length,
            nodes_count=nodes_count
        )
        
        self.dataset_length = len(self.dataset)
        
    def get_data_window(self, idx: int) -> torch.Tensor:
        '''
        Get the data window from the dataset.
        
        Parameters:
            - idx (int): Index of the data window.
            
        Returns:
            - torch.Tensor: Data window.
        '''
        return self.dataset.__getitem__(idx)
    
    def get_entire_time_series(self) -> np.ndarray:
        '''
        Get the time series from the dataset.
        
        Returns:
            - np.ndarray: Time series.
        '''
        return self.dataset.get_time_series()
 
  
