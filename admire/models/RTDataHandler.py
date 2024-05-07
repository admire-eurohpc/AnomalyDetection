import pandas as pd
import numpy as np
import redis
import time
import os
import torch.multiprocessing

from datetime import datetime
from typing import Dict, List
import os
import wandb
import pickle

from timeseriesdatasetv2 import TimeSeriesDatasetv2
from RTMetricsEvaluator import RTMetricsEvaluator

from utils.transformations import Transform




class RTDataHandler:
    def __init__(self, 
            data_dir: str, 
            inference_model_config: dict[str, str],
            debug_printing: bool = False
        ) -> None:
        '''
        For simulation purposes we need to check time of invoking RTDataHandler to read/save proper data.
        We don't need year, month, day information since it's only one day of data passing through
        Intentionally omitting seconds, cause you can easily find required data through hour and minute values

        Simulation performed assuming history starting at current_hour - self.batch_time
        History lenght : 1 day
        Simulated_db_data length : 1 day
        '''
        #For now implementing second option
        current_time = datetime.now()
        self.r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.hour = current_time.hour
        self.minute = current_time.minute
        self.batch_time = 5 #TODO : remove hardcoding, it would be best to include this information and number of nodes in config data provided to RTDataHandler
        
        # General data directory
        self.data_dir = data_dir
        self.debug_printing = debug_printing
        self.metrics_evaluator = RTMetricsEvaluator(model_config=inference_model_config, print_debug=debug_printing)
        
        # Get the transform object
        self.wandb_entity = inference_model_config['entity']
        self.wandb_project = inference_model_config['project']
        self.wandb_run_id = inference_model_config['run_id']
        
        # Get the transform object
        self.__get_transform()

    def log_to_db(self, nodenames: List, metrics: Dict) -> None:
        '''
        Logs to redis information provided by RTMetricsEvaluator.
        We are logging 2 information per node : 
        - anomaly_detected (bool)
        - z-score or some other metric indicating deviation from group mean

        For now abandoning time-series approach as it needs newest glibc which may not be present at Turin Cluster
        '''
        redis_data = dict([nodename, recon_error] for nodename, recon_error in zip(nodenames, metrics['per_node_metrics_df']['r_err']))
        self.r.hmset('anomaly-detection', redis_data)
    
    def get_new_data_from_db(self) -> np.array:
        '''
        Default pipeline for feeding data every n minutes.
        TODO: How to ensure reading proper data when script is automatically started every n minutes?
        Desired solution : 
        - Cron starting every 5 minutes the bash script with RTDataHandler
        - Each time Cron invokes script, it reads first n minutes of data and then cuts it off
        '''
        db_dataloader = TimeSeriesDatasetv2(
            data_dir=f'{self.data_dir}/valid_data',
            normalize=True,
            external_transform=self.transform,
            window_size=self.batch_time,
            slide_length=1,
            nodes_count=200
        )
        data_batch = db_dataloader[0][0].numpy()

        return data_batch
    
    def trim_db_data(self):
        '''
        Cuts first n minutes of data, so each time Cron invokes the script, first n minutes of data are the proper ones to read
        '''
        db_dataloader = TimeSeriesDatasetv2(
            data_dir=f'{self.data_dir}/valid_data',
            normalize=True,
            external_transform=self.transform,
            window_size=self.batch_time,
            slide_length=1,
            nodes_count=200
        )
        data_batch = db_dataloader.get_time_series()
        node_names = db_dataloader.get_node_names()
        dates_range = db_dataloader.get_dates_range()
        dates_range['start'] = dates_range['start'].replace(minute=(dates_range['start'].minute+self.batch_time)%60)
        data_batch_new = data_batch[:,:,self.batch_time:]
        self.save_dataset(data_batch_new, node_names, dates_range=dates_range, dir=os.path.join(self.data_dir, 'valid_data_updated'))


    def get_history(self) -> tuple[np.array, List, Dict[str, datetime]]:
        history_dataloader = TimeSeriesDatasetv2(
            data_dir=f'{self.data_dir}/history',
            normalize=True,
            external_transform=self.transform,
            window_size=self.batch_time,
            slide_length=1,
            nodes_count=200
        )
        history = history_dataloader.get_time_series()
        node_names = history_dataloader.get_node_names()
        dates_range = history_dataloader.get_dates_range()

        #Setting history dates_range to batch_time minutes before so with addition of current batch we get current time
        dates_range['start'] = dates_range['start'].replace(hour=self.hour, minute=(self.minute-self.batch_time)%60)
        dates_range['end'] = dates_range['end'].replace(hour=self.hour, minute=(self.minute-self.batch_time)%60)

        return history, node_names, dates_range
    
    def __get_transform(self) -> Transform:
        '''
        Get the Transform object from wandb
        '''
        # Set up the API
        api = wandb.Api()
        run = api.run(f"{self.wandb_entity}/{self.wandb_project}/{self.wandb_run_id}")
        
        # Get the artifact and download the model to local directory
        artifact = api.artifact(f'{self.wandb_entity}/{self.wandb_project}/Transform:v0', type='object')
        path = artifact.download()
        path = os.path.join(path, 'transform.pkl')
        
        with open(path, 'rb') as f:
            self.transform: Transform = pickle.load(f)
    
    def save_dataset(self, history: np.array, node_names: List, dates_range: Dict[str, datetime], dir: str) -> pd.DataFrame:
        for elem, filename in zip(history, node_names):
            _df = pd.DataFrame(np.transpose(elem), columns = ['power', 'cpu1', 'cpu2', 'cpus_alloc'])
            _df['date'] = pd.date_range(dates_range['start'], dates_range['end'], freq='1min', tz='UTC')
            _df.to_feather(os.path.join(dir, filename + '.feather'))
    
    def get_metrics_from_db(self, ) -> dict:
        '''
        We could check for consistency of our detection and check against previous detections.
        '''
        pass

    def append_log_file(self, ) -> None:
        '''
        If we want to check against previous detection, we need to update log_file every data batch
        '''
        pass

    def data_checkup(self) -> None:
        '''
        Some tests to run against incoming data? Maybe it's redundant in a demo, but for sure will be needed in real life application.
        '''
        pass

    def caluclate_metrics(self, metric: str = 'L1') -> dict:
        '''
        Metrics Evaluator for short data batches.
        '''
        metrics_dict = self.metrics_evaluator.calculate_metrics_for_last_window(rec_metric=metric)
        
        if self.debug_printing:
            print(f'Calculated metrics: {metrics_dict}')
        
        return metrics_dict
        
        

    def run(self,) -> None:
        torch.multiprocessing.set_sharing_strategy('file_system')
        start = time.time()

        #get simulated db data
        data_batch = self.get_new_data_from_db()

        #get history data
        history, node_names, dates_range = self.get_history()

        history_new = np.concatenate((history, data_batch), axis=2)

        dates_range_new={}
        dates_range_new['start'] = dates_range['start'].replace(hour=self.hour, minute=self.minute)
        dates_range_new['end'] = dates_range['end'].replace(hour=self.hour, minute=self.minute)

        self.save_dataset(history_new[:,:,self.batch_time:], node_names, dates_range_new, os.path.join(self.data_dir, 'history_updated'))
        
        self.trim_db_data()

        metrics = self.caluclate_metrics()

        self.log_to_db(node_names, metrics)

        end = time.time()
        print(end-start)

if __name__ == '__main__':
    
    test_data = 'data/processed/turin_demo_top200'
    
    model_config = {
        'model': 'LSTMCNN',
        'run_id': 'e_-2024_04_10-11_23_52',
        'entity': "ignacysteam",
        'project': "lightning_logs",
        'model_tag': "v0",
        'data_dir': 'data/processed/turin_demo_top200/history',
        'data_normalization': True,
        'slide_length': 1,
        'nodes_count': 200
    }
    
    #For testing purposes feeding date from bash script
    logger = RTDataHandler(data_dir=test_data, inference_model_config=model_config, debug_printing=True)
    logger.run()
