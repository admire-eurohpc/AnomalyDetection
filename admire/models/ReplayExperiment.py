import numpy as np
import copy

from RT_model_wrapper import ModelLoaderService, ModelInference, DataLoaderService


class ReplayExperiment:
    
    model_service: ModelLoaderService = None
    model_inference: ModelInference = None
    data_loader: DataLoaderService = None
    
    def __init__(self, experiment_config: dict[str, str], print_debug: bool = True) -> None:
        self.experiment_config = experiment_config
        
        # Create the model loader service
        self.model_service = ModelLoaderService(experiment_config['model'])
        self.model_service.load_model_from_wandb(run_id=experiment_config['run_id'])
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
            data_dir=experiment_config['data_dir'],
            data_normalization=experiment_config['data_normalization'],
            window_size=window_size,
            slide_length=experiment_config['slide_length'],
            nodes_count=experiment_config['nodes_count']
        )
    
    def replay_entire_time_series(self, print_debug: bool = True) -> np.ndarray:
        '''
        Replay the entire time series.
        
        Returns:
            - np.ndarray: Inferred data.
        '''
        length = self.data_loader.dataset_length
        channels = self.data_loader.dataset.time_series.shape[1]
        nodes = self.data_loader.dataset.time_series.shape[0]
        
        metrics_per_windowL1 = {
            'per_sample_and_channel': np.empty(shape=(length, nodes, channels)),
            'per_sample': np.empty(shape=(length, nodes)),
            'per_batch': np.empty(shape=(length,))
        }
        metrics_per_windowL2 = copy.deepcopy(metrics_per_windowL1)
        
        for i in range(length):
            data = self.data_loader.get_data_window(i)
            
            if print_debug:
                print(f"Replaying window {i}... Data shape: {data.shape}")
            
            # inferred_data = np.empty_like(data)
            
            # for j in range(nodes):
            #     _data = data[j].unsqueeze(0)
            #     _inferred_data = self.model_inference.infer(_data)
            #     inferred_data[j] = _inferred_data
            
            inferred_data = self.model_inference.infer(data)
            original_data = data.cpu().numpy()
            
            # Calculate the reconstruction error
            errorL1_dict = self.model_inference.calculate_reconstruction_error(original_data, inferred_data, metric='L1')
            errorL2_dict = self.model_inference.calculate_reconstruction_error(original_data, inferred_data, metric='L2')

            # Save the metrics
            metrics_per_windowL1['per_sample_and_channel'][i] = errorL1_dict['per_sample_and_channel']
            metrics_per_windowL1['per_sample'][i] = errorL1_dict['per_sample']
            metrics_per_windowL1['per_batch'][i] = errorL1_dict['per_batch']
            
            metrics_per_windowL2['per_sample_and_channel'][i] = errorL2_dict['per_sample_and_channel']
            metrics_per_windowL2['per_sample'][i] = errorL2_dict['per_sample']
            metrics_per_windowL2['per_batch'][i] = errorL2_dict['per_batch']
                
            # TODO: This is just for debug. Remove it later
            if i == 10:
                break
    
        return {
            'L1': metrics_per_windowL1,
            'L2': metrics_per_windowL2
        }
          
    
if __name__ == '__main__':    
    experiment_config = {
        'model': 'LSTMCNN',
        'run_id': 'e_standard-2024_02_20-14_08_29',
        'data_dir': 'data/processed/train_up_to_december_test_feb_to_march_200nodes/test',
        'data_normalization': True,
        'slide_length': 1,
        'nodes_count': 200
    }
    
    replay = ReplayExperiment(experiment_config, print_debug=True)
    metrics = replay.replay_entire_time_series()
    
    print(metrics)