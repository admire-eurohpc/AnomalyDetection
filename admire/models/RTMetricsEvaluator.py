import numpy as np
import pandas as pd
import itertools
import tqdm

import torch.multiprocessing as mp
import lightning.pytorch as pl

from utils.metrics_evaluation import MetricsEvaluator
from RT_model_wrapper import ModelLoaderService, ModelInference, DataLoaderService, Prepare_distributed_data

MINUTES_TO_USE = 360
LONG_WINDOW_SIZE = 180
SHORT_WINDOW_SIZE = 60

class RTMetricsEvaluator:
    
    metrics_evaluator: MetricsEvaluator = None
    model_service: ModelLoaderService = None
    model_inference: ModelInference = None
    data_loader: DataLoaderService = None
    
    def __init__(self, model_config: dict[str, str], 
            print_debug: bool = False,
            long_window_size: int = LONG_WINDOW_SIZE,
            short_window_size: int = SHORT_WINDOW_SIZE
        ) -> None:
        '''
        Initialize the RTMetricsEvaluator.
        
        Parameters:
            - model_config (dict[str, str]): The model configuration dictionary.
            - print_debug (bool): Whether to print debug information. Default is False.
            - long_window_size (int): The size of the long window. Default is 180.
            - short_window_size (int): The size of the short window. Default is 60.
        '''
        self.model_config = model_config
        self.print_debug = print_debug
        
        self.long_window_size = long_window_size
        self.short_window_size = short_window_size
        
        # Create the model loader service
        self.model_service = ModelLoaderService(model_config['model'])
        self.model_service.load_model_from_wandb(
            run_id=model_config['run_id'],
            entity=model_config['entity'],
            project=model_config['project'],
            model_tag=model_config['model_tag']
        )
        
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
        self.window_size = int(model_hparams['window_size'])
        
        self.data_loader = DataLoaderService(
            data_dir=model_config['data_dir'] + '/history',
            data_normalization=model_config['data_normalization'],
            window_size=self.window_size,
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
        nc_rec_error, n_rec_error, t_rec_error = self.__obtain_reconstruction_error_for_a_window(data, rec_metric)
        
        # Calculate the metrics, each row is a different node
        per_node_df = pd.DataFrame(n_rec_error, columns=['r_err'], index=range(len(n_rec_error)))
        
        # Add the channel reconstruction error
        channels = nc_rec_error.shape[1]
        channels_cols = [f'channel_{i}_r_err' for i in range(channels)]
        per_node_df[channels_cols] = nc_rec_error
        
        # Calculate the sliding window metrics
        sliding_window_metrics = self.__get_metrics_using_sliding_window(rec_metric)
        
        # Create the return dictionary
        return_dict = {
            'per_node_metrics_df': per_node_df,
            'total_rec_error': t_rec_error
        }
        
        # Add the sliding window metrics
        return_dict = {**return_dict, **sliding_window_metrics}
        
        return return_dict
    
    def __get_metrics_using_sliding_window(self, rec_metric: str = 'L1') -> dict:
        # Calculate the sliding window mean for the long and short windows
        comprehensive_metrics = self.__obtain_reconstruction_error_for_entire_time_series(rec_metric=rec_metric)
        time_series = comprehensive_metrics['per_batch']
        sliding_window_mean_long = MetricsEvaluator.average_past_window(time_series, self.long_window_size)
        sliding_window_mean_short = MetricsEvaluator.average_past_window(time_series, self.short_window_size)


        # Amputate the first long_window_size elements
        sliding_window_mean_long = sliding_window_mean_long[:len(sliding_window_mean_long)-self.long_window_size]
        # Amputate the first short_window_size elements
        sliding_window_mean_short = sliding_window_mean_short[:len(sliding_window_mean_short)-self.long_window_size]

        difference = sliding_window_mean_short - sliding_window_mean_long 
        
        # Average difference over the last day
        avg = np.mean(difference)
        std = np.std(difference)
        
        # Check if the difference is significant in the last window ie we think that there exi
        last_diff = difference[-1]
        last_diff_significant = False
        if abs(last_diff) > avg + 2 * std:
            last_diff_significant = True
            
        # Check which nodes contribute the most to the difference
        pc_sliding_window_mean_long = MetricsEvaluator.average_past_window_per_node(comprehensive_metrics['per_sample'], self.long_window_size)
        pc_sliding_window_mean_short = MetricsEvaluator.average_past_window_per_node(comprehensive_metrics['per_sample'], self.short_window_size)
        pc_sliding_window_mean_long = pc_sliding_window_mean_long[:len(pc_sliding_window_mean_long)-self.long_window_size]
        pc_sliding_window_mean_short = pc_sliding_window_mean_short[:len(pc_sliding_window_mean_short)-self.long_window_size]
        
        # Calculate the difference per channel
        pc_difference = pc_sliding_window_mean_short - pc_sliding_window_mean_long
        pc_avg = np.mean(pc_difference, axis=0)
        pc_std = np.std(pc_difference, axis=0)
        
        # Check if the difference is significant in the last window ie we think that there exi
        pc_last_diff = pc_difference[-1]
        pc_sum = np.sum(pc_last_diff)
        pc_last_diff_normalized = pc_last_diff / pc_sum # Now each element is the contribution of the node to the difference in percentage
        
        pc_last_diff_significant = np.abs(pc_last_diff) > pc_avg + 2 * pc_std
        
        
        return {
            'last_window_is_anomaly': last_diff_significant,
            'last_window_is_anomaly_per_node': pc_last_diff_significant,
            'nodes_contribution_to_difference': pc_last_diff,
            'nodes_contribution_to_difference_normalized': pc_last_diff_normalized,
        }
    
    # TODO: Not used in the current implementation
    def __calculate_auxillary_metrics_for_entire_time_series(self, rec_metric: str = 'L1') -> dict:
        
        # Calculate the mean and std of the reconstruction error for the entire time series
        comprehensive_metrics = self.__obtain_reconstruction_error_for_entire_time_series(rec_metric)
        
        per_node= comprehensive_metrics['per_sample']
        timestep_mean = np.mean(per_node, axis=0)
        timestep_std = np.std(per_node, axis=0)
        
        per_node_and_channel = comprehensive_metrics['per_sample_and_channel']
        per_node_and_channel_mean = np.mean(per_node_and_channel, axis=0)
        per_node_and_channel_std = np.std(per_node_and_channel, axis=0)
        
        total_rec_error = comprehensive_metrics['per_batch']
        total_rec_error_mean = np.mean(total_rec_error)
        total_rec_error_std = np.std(total_rec_error)
        
        return {
            'timestep_mean': timestep_mean,
            'timestep_std': timestep_std,
            'timestep_per_channel_mean': per_node_and_channel_mean,
            'timestep_per_channel_std': per_node_and_channel_std,
            'total_rec_error_mean': total_rec_error_mean,
            'total_rec_error_std': total_rec_error_std
        }
    

    def multiprocess_recon_error(self, sample):
        _iter, window_idx = sample
        original_data = self.data_loader.get_data_window(window_idx)[0].cpu().detach().numpy()
        inferred_data = self.outputs[_iter]
        errorL1_dict = self.model_inference.calculate_reconstruction_error(original_data, inferred_data, metric='L1')
        return _iter, errorL1_dict
    

    def __obtain_reconstruction_error_for_entire_time_series(self, minutes_to_use: int = MINUTES_TO_USE, rec_metric: str = 'L1') -> dict:
        '''
        Obtain the reconstruction error for the entire time series.
        
        Parameters:
            - minutes_to_use (int): The number of minutes to use for the calculation of historic metrics, that is used to calculate the sliding window mean.
                These minutes are counted from the end of the time series. Default is 360 i.e, 6 hours.
            - rec_metric (str): The reconstruction metric to use. Default is 'L1'. Options are 'L1' and 'L2'.
        
        Returns:
            - dict: The metrics dictionary.
        '''
        assert minutes_to_use > 0, 'minutes_to_use must be greater than 0.'
        
        length = self.data_loader.dataset_length # The length of the entire time series
        channels = self.data_loader.dataset.time_series.shape[1]
        nodes = self.data_loader.dataset.time_series.shape[0]
        
        assert length - minutes_to_use >= 0, 'minutes_to_use must be less than the length of the time series.'
        
        shortened_length = length - minutes_to_use # The length of the time series to use for the calculation of historic metrics

        comprehensive_metrics = {
            'per_sample_and_channel': np.empty(shape=(shortened_length, nodes, channels)),
            'per_sample': np.empty(shape=(shortened_length, nodes,)),
            'per_batch': np.empty(shape=(shortened_length,))
        }


        model = self.model_service.get_model()


        data = Prepare_distributed_data(data_dir=self.model_config['data_dir']+'/history',
            data_normalization=self.model_config['data_normalization'],
            window_size=self.window_size,
            slide_length=self.model_config['slide_length'],
            nodes_count=self.model_config['nodes_count'],)


        trainer = pl.Trainer(accelerator='cpu', strategy='ddp', devices=4, callbacks=[], use_distributed_sampler=False)
        trainer.predict(model=model, dataloaders=data, return_predictions=False)
        if trainer.global_rank == 0:
            outputs, idxs = model.on_predict_end()
            idxs = list(itertools.chain.from_iterable([y.tolist() for y in idxs]))
            outputs = [x for (y,x) in sorted(zip(idxs,outputs), key=lambda pair: pair[0])]

            excess_sample_nb = len(outputs) - len(data)

        if trainer.global_rank > 0:
            exit(0)

        self.outputs = outputs[:-excess_sample_nb]

        with mp.Pool(mp.cpu_count()) as pool:
            for _iter, errorL1_dict in tqdm.tqdm(pool.imap(self.multiprocess_recon_error, enumerate(range(length - minutes_to_use, length)), chunksize=20),
                                    desc="Running inference reconstruction error", 
                                    total=len(range(length - minutes_to_use, length))):
                comprehensive_metrics['per_sample_and_channel'][_iter] = errorL1_dict['per_sample_and_channel']
                comprehensive_metrics['per_sample'][_iter] = errorL1_dict['per_sample']
                comprehensive_metrics['per_batch'][_iter] = errorL1_dict['per_batch']
        
        print("Calculating recon_error done")

        return comprehensive_metrics
        
    def __obtain_reconstruction_error_for_a_window(self, data: np.ndarray, rec_metric: str = 'L1') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        data = data[0]
        inferred_data = self.model_inference.infer(data)
        
        error_dict =  self.model_inference.calculate_reconstruction_error(data, inferred_data, metric=rec_metric)
        
        per_node_and_channel_rec_error = error_dict['per_sample_and_channel']
        per_node_rec_error = error_dict['per_sample']
        total_rec_error = error_dict['per_batch']
        
        return per_node_and_channel_rec_error, per_node_rec_error, total_rec_error
        
        