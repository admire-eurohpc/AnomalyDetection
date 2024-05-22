import numpy as np
import numpy.typing as npt
import logging


class Transform:
    '''
    Base class for transformations
    '''
    
    def __init__(self) -> None:
        pass
    
    def fit(self, time_series: npt.NDArray, channel_dim: int = 2) -> None:
        '''
        Fit the transformation to the data
        
        `time_series`: Time series to fit the transformation to
        `channel_dim`: Dimension of the channels, default is 0
        '''
        self.min = np.min(np.min(time_series, axis=channel_dim, keepdims=True), axis=1, keepdims=True)
        self.max = np.max(np.max(time_series, axis=channel_dim, keepdims=True), axis=1, keepdims=True)
        logging.debug(f"Min: {list(self.min)}")
        logging.debug(f"Max: {list(self.max)}")

    
    def normalize_time_series(self, time_series: npt.NDArray) -> npt.NDArray:
        '''
        Normalizes a time series by min-max \n
        Requires that the transformation has been fit previously
        
        `time_series`: Time series to normalize
        
        Return: Normalized time series as numpy array
        '''
        return (time_series - self.min) / (self.max - self.min)
    
    def denormalize_time_series(self, time_series: npt.NDArray) -> npt.NDArray:
        '''
        Reverts the normalization of a time series. 
        Requires that the transformation has been fit previously
        
        `time_series`: Time series to revert normalization of
        
        Return: Reverted normalized time series as numpy array
        '''
        _max = self.max.flatten()
        _min = self.min.flatten()
        
        # Rearrange time_series
        _time_series = time_series.transpose(0, 2, 1)
        
        _time_series = _time_series * (_max - _min) + _min
        
        # Rearrange back
        return _time_series.transpose(0, 2, 1)
    
    def serialize(self) -> dict:
        '''
        Serialize the transformation
        
        Return: Dictionary with the transformation
        '''
        return {
            'min': self.min,
            'max': self.max
        }
    
    def create_from_dict(self, data: dict) -> None:
        '''
        Create a transformation from a dictionary
        
        `data`: Dictionary with the transformation
        '''
        self.min = data['min']
        self.max = data['max']