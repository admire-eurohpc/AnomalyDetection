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
import matplotlib.pyplot as plt
import configparser

from utils.transformations import Transform

logger = logging.getLogger(__name__)
config = configparser.ConfigParser()
config.read('config.ini')

class TimeSeriesDataset(Dataset):
    '''Dataset generator class for time series data'''
    def __init__(self, data_dir: str, 
                 transform=None, 
                 target_transform=None, 
                 normalize: bool = True,
                 window_size: int = 60,
                 slide_length: int = 1,
                 external_transform: Transform = None
                 ) -> None:
        '''
        `data_dir`: Directory where the data is stored
        `transform`: Transform to apply to the data [NOT IMPLEMENTED]
        `target_transform`: Transform to apply to the target [NOT IMPLEMENTED]
        `normalize`: Whether to normalize the data
        `window_size`: Size of the window to use for the time series
        `slide_length`: How many time steps to slide the window by
        `external_transform`: If you want to use a pre-fitted transform, pass it here
        '''

        self.time_series: npt.NDarray = None
        self.filenames = []
        self.window_size = window_size
        self.slide_length = slide_length
        self.transform = transform
        self.target_transform = target_transform
        self.normalize = normalize
        self.include_cpu_alloc = config.getboolean('PREPROCESSING','with_cpu_alloc')
        self.nodes_count = int(config['PREPROCESSING']['nodes_count_to_process'])
        
        # Get all filenames in data_dir
        _, _, filenames = os.walk(data_dir).__next__()
        logger.debug(f"Found {len(filenames)} files in {data_dir}. These are: {filenames}")
        
        # Sort filenames to ensure they are in order
        # TODO: Later this should be made more robust so we always know the order and which nodes are which
        filenames.sort()

        self.dates_range = {}
        _dates = pd.read_parquet(os.path.join(data_dir, filenames[0]), columns=['date'])
        self.dates_range['start'] = pd.to_datetime(_dates['date'].min())
        self.dates_range['end'] =  pd.to_datetime(_dates['date'].max())

        # Concatenate data into one time series array (numpy array)
        # TODO: This is not efficient as for bigger datasets may load too much data into memory
        # but it works for now
        # Desired shape will be (n_features x n_nodes x n_time_steps )
        # e.g. (3 features x 100 nodes x 1000 time ticks )
        for filename in tqdm.tqdm(filenames, desc="Loading ts data"):
            columns = ['power', 'cpu1', 'cpu2']
            if self.include_cpu_alloc:
                columns.append('cpus_alloc')
            _data = pd.read_parquet(
                        os.path.join(data_dir, filename), 
                        columns=columns
                        ) \
                    .to_numpy().T.reshape(len(columns), 1, -1) 
                    
            #logger.debug(f'Loaded data shape: {_data.shape}')
                    
            if self.time_series is None:
                self.time_series = _data
            else:
                self.time_series = np.concatenate((self.time_series, _data), axis=2)

            self.filenames.append(filename[0:5])
        logger.debug(f"Time series shape after concatenation: {self.time_series.shape}")
        
        if self.normalize and not external_transform:
            logger.info("Normalizing time series")
            self.transform = Transform()
            self.transform.fit(self.time_series)
            self.time_series = self.transform.normalize_time_series(self.time_series)
        elif self.normalize and external_transform: # Useful for testing
            logger.info("Normalizing time series with external transform")
            self.transform = external_transform
            self.time_series = self.transform.normalize_time_series(self.time_series)
                
        # It is important to convert to float32, otherwise pytorch will complain
        self.time_series = self.time_series.astype(np.float32)
        self.time_series = np.reshape(self.time_series, (self.time_series.shape[1], self.time_series.shape[0], self.time_series.shape[2]))
        
        logger.debug(f"Time series shape: {self.time_series.shape}")


    def __len__(self):
        '''
        Returns the number of windows in the time series given the window size and slide length
        '''
        return ((self.time_series.shape[2] - self.window_size)// self.slide_length) + 1 # TODO: Check this thoroughly

    def __getitem__(self, idx): 
        start = idx * self.slide_length # Each window starts at a multiple of the slide length
        ts = self.time_series[:, :, start:start+self.window_size] # Get the window
        return torch.Tensor(ts)
    
    def get_time_series(self):
        '''Returns the time series. If normalized, returns the denormalized time series'''
        # if self.normalize:
        #     return self.transform.denormalize_time_series(self.time_series)
        
        return self.time_series
    
    def get_node_len(self):
        '''Returns the node length adjusted to dataloader scenario (without final N samples) 
         since we can't reconstruct N min window depending on less than N samples'''
        return ((self.time_series.shape[2]//self.nodes_count - self.window_size)// self.slide_length) + 1
    
    def get_node_full_len(self):
        '''Returns the full node length including last N samples'''
        return self.time_series.shape[2]//self.nodes_count 
    
    def get_input_layer_size_flattened(self):
        return self.time_series.shape[1] * self.time_series.shape[2] * self.window_size
    
    def get_input_layer_shape(self):
        '''(n_features x n_nodes x n_time_steps)'''
        return self.time_series.shape[0], self.time_series.shape[1], self.window_size
    
    def get_transform(self):
        '''Returns the transform object used to normalize the data'''
        return self.transform
    
    def get_filenames(self):
        return self.filenames
    
    def get_dates_range(self) -> Dict[str, datetime]:
        '''Returns the start and end dates of the time series'''
        return self.dates_range
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    dataset = TimeSeriesDataset(data_dir="data/processed/march22-24_top200_withalloc_and_augm_fixed_hours/test", normalize=True)
    d = next(iter(dataset))
    print(d.shape)
    print(d)
    