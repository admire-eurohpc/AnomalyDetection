import os
import pandas as pd
import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset
import logging
import tqdm

logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    '''Dataset generator class for time series data'''
    def __init__(self, data_dir: str, 
                 transform=None, 
                 target_transform=None, 
                 window_size: int = 20,
                 slide_length: int = 10,
                 ) -> None:
        '''
        `data_dir`: Directory where the data is stored
        `transform`: Transform to apply to the data [NOT IMPLEMENTED]
        `target_transform`: Transform to apply to the target [NOT IMPLEMENTED]
        `window_size`: Size of the window to use for the time series
        `slide_length`: How many time steps to slide the window by
        '''

        self.time_series: npt.NDarray = None
        self.window_size = window_size
        self.slide_length = slide_length
        self.transform = transform
        self.target_transform = target_transform
        
        # Get all filenames in data_dir
        _, _, filenames = os.walk(data_dir).__next__()
        logger.debug(f"Found {len(filenames)} files in {data_dir}. These are: {filenames}")

        # Concatenate data into one time series array (numpy array)
        # TODO: This is not efficient, but it works for now
        # Desired shape will be (n_nodes x n_time_steps x n_features)
        # e.g. (100 nodes x 1000 time ticks x 3 features)
        for filename in tqdm.tqdm(filenames, desc="Loading ts data"):
            _data = pd.read_parquet(
                        os.path.join(data_dir, filename), 
                        columns=['power', 'cpu1', 'cpu2']
                        ) \
                    .to_numpy() \
                    .reshape(1, -1, 3)
                    
            if self.time_series is None:
                self.time_series = _data
            else:
                self.time_series = np.concatenate((self.time_series, _data), axis=0)
                
        # It is important to convert to float32, otherwise pytorch will complain
        self.time_series = self.time_series.astype(np.float32)
        
        # Reshape time series so the first dimension is the time dimension
        # (n_time_steps x n_nodes x n_features)
        self.time_series = self.time_series.reshape(self.time_series.shape[1], self.time_series.shape[0], self.time_series.shape[2])
            
        logger.debug(f"Time series shape: {self.time_series.shape}")
        logger.debug(f'Time series sample: {self.time_series[0:3, 0:3, :]}')


    def __len__(self):
        '''
        Returns the number of windows in the time series given the window size and slide length
        '''
        return len(self.time_series) // self.slide_length - self.window_size // self.slide_length # TODO: Check this thoroughly

    def __getitem__(self, idx): 
        start = idx * self.slide_length # Each window starts at a multiple of the slide length
        ts = self.time_series[start:start+self.window_size, :, :] # Get the window
        return ts
    
    def get_time_series(self):
        return self.time_series
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    dataset = TimeSeriesDataset(data_dir="data/processed/")

    d = next(iter(dataset))
    print(d.shape)
    print(d)
    