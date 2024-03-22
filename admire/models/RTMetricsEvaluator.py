import numpy as np
import pandas as pd

from utils.metrics_evaluation import MetricsEvaluator
from RT_model_wrapper import ModelLoaderService, ModelInference, DataLoaderService

class RTMetricsEvaluator:
    
    metrics_evaluator: MetricsEvaluator = None
    model_service: ModelLoaderService = None
    model_inference: ModelInference = None
    data_loader: DataLoaderService = None
    
    def __init__(self, model_config: dict[str, str], print_debug: bool = False) -> None:
        self.model_config = model_config
        
        # Create the model loader service
        self.model_service = ModelLoaderService(model_config['model'])
        self.model_service.load_model_from_wandb(run_id=model_config['run_id'])
        if print_debug:
            self.model_service.print_model_summary()
            
        # Create the model inference service
        self.model_inference = ModelInference(
            model=self.model_service.get_model(),
            model_type=self.model_service.get_model_type(),
            hyperparameters=self.model_service.hyperparameters
        )
        
        # Create the data loader service
        model_hparams = self.model_service.hyperparameters
        window_size = int(model_hparams['window_size'])
        
        self.data_loader = DataLoaderService(
            data_dir=model_config['data_dir'],
            data_normalization=model_config['data_normalization'],
            window_size=window_size,
            slide_length=model_config['slide_length'],
            nodes_count=model_config['nodes_count']
        )
        
    def calculate_metrics_for_last_window(self, rec_metric: str = 'L1') -> dict:
        '''
        Calculate the metrics for the  last window of data.
        
        Parameters:
            - rec_metric (str): The reconstruction metric to use. Default is 'L1'. Options are 'L1' and 'L2'.
        
        Returns:
            - dict: The metrics dictionary.
        '''
        
        length = self.data_loader.dataset_length
        last_window_idx = length - 1
        
        data = self.data_loader.get_data_window(last_window_idx)
        
        # First obtain the reconstruction error
        nc_rec_error, n_rec_error, t_rec_error = self.obtain_reconstruction_error(data, rec_metric)
        
        # Calculate the metrics, each row is a different node
        per_node_df = pd.DataFrame(n_rec_error, columns=['r_err'], index=range(len(n_rec_error)))
        
        # Add the channel reconstruction error
        channels = nc_rec_error.shape[1]
        per_node_df[[f'channel_{i}_r_err' for i in range(channels)]] = nc_rec_error
        
        return_dict = {
            'per_node_metrics_df': per_node_df,
            'total_rec_error': t_rec_error
        }
        
        return return_dict
        
        
    def obtain_reconstruction_error(self, data: np.ndarray, rec_metric: str = 'L1') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Obtain the reconstruction error for the given data.
        
        Parameters:
            - data (np.ndarray): The data to evaluate.
            - rec_metric (str): The reconstruction metric to use. Default is 'L1'. Options are 'L1' and 'L2'.
        
        Returns:
            -  tuple[np.ndarray, np.ndarray, np.ndarray]: The per node and channel reconstruction error, 
                the per node reconstruction error, and the total reconstruction error.
        '''
        assert rec_metric in ['L1', 'L2'], 'rec_metric must be either "L1" or "L2".'
        
        inferred_data = self.model_inference.infer(data)
        
        error_dict =  self.model_inference.calculate_reconstruction_error(data, inferred_data, metric=rec_metric)
        
        per_node_and_channel_rec_error = error_dict['per_sample_and_channel']
        per_node_rec_error = error_dict['per_sample']
        total_rec_error = error_dict['per_batch']
        
        return per_node_and_channel_rec_error, per_node_rec_error, total_rec_error
        
        