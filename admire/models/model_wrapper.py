from ae_encoder import CNN_encoder, CNN_LSTM_encoder
from ae_decoder import CNN_decoder, CNN_LSTM_decoder
from ae_litmodel import LitAutoEncoder, LSTM_AE
from ae_lstm_vae import LSTMVAE, LSTMEncoder, LSTMDecoder

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
    
    def __init__(self, model_type: str):
        '''
        Parameters:
            - model_type (str): Type of the model to be used. 
        '''
        assert model_type.upper() in self.supported_model_types, \
            f"Model type {model_type} not supported. Supported model types: {self.supported_model_types}"
            
        self.model_type = model_type.upper()
    
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
  
if __name__ == '__main__':
    model_service = ModelLoaderService('LSTMCNN')
    model_service.load_model_from_wandb(run_id='e_standard-2024_02_20-14_08_29')
    model_service.print_model_summary()
    