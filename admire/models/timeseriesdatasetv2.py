from datetime import datetime
import os
from typing import Dict
import pandas as pd
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset
import logging
import tqdm

from utils.transformations import Transform

class TimeSeriesDatasetv2(Dataset):
    '''Dataset generator class for time series data'''
    def __init__(self, data_dir: str, 
                 nodes_count: int,
                 transform=None, 
                 target_transform=None, 
                 normalize: bool = True,
                 window_size: int = 60,
                 slide_length: int = 1,
                 external_transform: Transform = None,
                 ) -> None:
        '''
        `data_dir`: Directory where the data is stored
        `transform`: Transform to apply to the data [NOT IMPLEMENTED]
        `target_transform`: Transform to apply to the target [NOT IMPLEMENTED]
        `normalize`: Whether to normalize the data
        `window_size`: Size of the window to use for the time series
        `slide_length`: How many time steps to slide the window by
        `external_transform`: If you want to use a pre-fitted transform, pass it here
        `nodes_count`: Number of nodes to process. If None, it will use the value from the config file
        '''

        self.time_series: npt.NDarray = None
        self.filenames = []
        self.window_size = window_size
        self.slide_length = slide_length
        self.transform = transform
        self.target_transform = target_transform
        self.normalize = normalize
        self.nodes_count = nodes_count 
        
        # Get all filenames in data_dir
        _, _, filenames = os.walk(data_dir).__next__()
        logging.debug(f"Found {len(filenames)} files in {data_dir}. These are: {filenames}")
        
        # Sort filenames to ensure they are in order
        # TODO: Later this should be made more robust so we always know the order and which nodes are which
        filenames.sort()

        self.dates_range = {}
        _dates = pd.read_parquet(os.path.join(data_dir, filenames[0]), columns=['date'])
        self.dates_range['start'] = pd.to_datetime(_dates['date'].min(), utc=True)
        self.dates_range['end'] =  pd.to_datetime(_dates['date'].max(), utc=True)
        
        # Concatenate data into one time series array (numpy array)
        # TODO: This is not efficient as for bigger datasets may load too much data into memory
        # but it works for now
        # Desired shape will be (n_features x n_nodes x n_time_steps )
        # e.g. (3 features x 100 nodes x 1000 time ticks )
        for filename in tqdm.tqdm(filenames, desc="Loading ts data"):
            columns = ['power', 'cpu1', 'cpu2', 'cpus_alloc']
            _data = pd.read_parquet(
                        os.path.join(data_dir, filename), 
                        columns=columns
                        ) \
                    .to_numpy().T.reshape(len(columns), 1, -1) 
            if slide_length != 1:
                cutoff_value = np.shape(_data)[2]%self.slide_length
                _data = _data[:,:,:-cutoff_value]
            if self.time_series is None:
                self.time_series = _data
            else:
                self.time_series = np.concatenate((self.time_series, _data), axis=1)

            self.filenames.append(filename[0:5])
        logging.debug(f"Time series shape after concatenation: {self.time_series.shape}")
        print(f"Time series shape after concatenation: {self.time_series.shape}")
        
        if self.normalize and not external_transform:
            logging.info("Normalizing time series")
            self.transform = Transform()
            self.transform.fit(self.time_series)
            self.time_series = self.transform.normalize_time_series(self.time_series)
        elif self.normalize and external_transform: # Useful for testing
            logging.info("Normalizing time series with external transform")
            self.transform = external_transform
            self.time_series = self.transform.normalize_time_series(self.time_series)
                
        # It is important to convert to float32, otherwise pytorch will complain
        self.time_series = self.time_series.astype(np.float32)
        self.time_series = np.reshape(self.time_series, (self.time_series.shape[1], self.time_series.shape[0], self.time_series.shape[2]))
        
        logging.debug(f"Time series shape: {self.time_series.shape}")
        print(f"Time series shape: {self.time_series.shape}")

    def __len__(self):
        '''
        Returns the number of windows in the time series given the window size and slide length
        '''
        '''
        return self.nodes_count*self.get_node_full_len()
        The expression above is simpler and yields the same value as current solution, however it results in uncommon error
        https://github.com/autonomousvision/carla_garage/issues/12
        '''
        return ((self.time_series.shape[2] - self.window_size)// self.slide_length) + 1 # TODO: Check this thoroughly

    def __getitem__(self, idx): 
        start = idx * self.slide_length # Each window starts at a multiple of the slide length
        ts = self.time_series[:, :, start:start+self.window_size] # Get the window
        return torch.Tensor(ts)
    
    def get_time_series(self):
        '''Returns the time series. If normalized, returns the denormalized time series'''  
        return self.time_series
    
    def get_node_names(self):
        '''Return filenames in order of reading'''
        return self.filenames
    
    def get_dates_range(self) -> Dict[str, datetime]:
        '''Returns the start and end dates of the time series'''
        return self.dates_range
