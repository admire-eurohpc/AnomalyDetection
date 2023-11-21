import numpy as np 
import numpy.typing as npt
import pandas as pd 

def calculate_z_score(data: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    '''
    Calculates the z-score of the data.
    
    Args:
        data: data to calculate the z-score for, requires a 2D array (batch_size, time_steps)
        
    Returns:
        z-scored data with the same shape as the input data
    '''
    mean = np.mean(data, axis=1).reshape(-1, 1)
    std = np.std(data, axis=1).reshape(-1, 1)

    return np.abs((data - mean) / std)
    
def get_nth_percentile(data: np.ndarray[np.float64], percentile: int) -> np.ndarray[np.float64]:
    '''
    Calculates the nth percentile of the data.
    
    Args:
        data: data to calculate the nth percentile for, requires a 2D array (batch_size, time_steps)
        percentile: percentile to calculate
        
    Returns:
        nth percentile of the data for each row
    '''
    
    return np.percentile(data, percentile, axis=1)

def threshold_data_by_value(data: np.ndarray, thresh_value: float | np.ndarray) -> np.ndarray[np.int8]:
    '''
    Thresholds the data by a given value.
    
    Args:
        data: data to threshold, requires a 2D array (batch_size, time_steps)
        thresh_value: value to threshold the data with, can be a single value or an array of values (batch_size, 1)
        
    Returns:
        thresholded data
    '''
    # Check if thresh_value is a single value or an array
    # If it is an array then if its one dimensional reshape it to (batch_size, 1)
    if type(thresh_value) is np.ndarray and len(thresh_value.shape) == 1:
        thresh_value = thresh_value.reshape(-1, 1)
    
    return (data > thresh_value).astype(np.int8)

def batch_median(data: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    '''
    Calculates the median of each batch.
    
    Args:
        data: data to calculate the median for, requires a 2D array (batch_size, time_steps)
        
    Returns:
        median on each time step
    '''
    return np.median(data, axis=0)

def batch_mean_only_working_nodes(data_to_transform: np.ndarray[np.float64], cpu_alloc_data: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    '''
    Calculates the mean of each batch using only the working nodes.
    
    Args:
        data_to_transform: data to calculate the mean for, requires a 2D array (batch_size, time_steps)
        cpu_alloc_data: data to determine the working nodes, requires a 2D array (batch_size, time_steps)
        
    Returns:
        mean on each time step
    '''

    working_nodes_mask = cpu_alloc_data > 0
    
    ret = np.zeros(data_to_transform.shape[1])
    
    for i in range(data_to_transform.shape[1]):
        mask = working_nodes_mask[:, i]
        if mask.sum() == 0:
            ret[i] = 0
        else:
            ret[i] = np.mean(data_to_transform[mask, i])
            
    return ret
    

if __name__ == '__main__': 
    
    reconstruction_error = pd.read_parquet("lightning_logs/ae/AE_CNN_ONE_NODE/2023_11_03-11_14_45/eval-['2023-03-01', '2023-03-30']/recon_error.parquet")
    
    print(reconstruction_error.info())
    print(reconstruction_error.shape)
    
    # Test each function
    
    # Calculate z-score
    z_score = calculate_z_score(reconstruction_error.values)
    print(z_score.shape)
    
    # Get 99th percentile
    percentile = get_nth_percentile(reconstruction_error, 99)
    print(percentile.shape)
    
    # Threshold data
    thresholded_data = threshold_data_by_value(z_score, 2)
    print(thresholded_data.shape)
    
    # Threshold rec error by percentile
    thresholded_data = threshold_data_by_value(reconstruction_error, percentile)
    print(thresholded_data.shape)
    
    # Batch median
    _batch_median = batch_median(reconstruction_error)
    print(_batch_median.shape)
    
    # Test batch mean only working nodes
    dummy_rec_error = np.array([[1, 2, 3, 4,5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
    dummy_cpu_alloc = np.array([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    
    ret = batch_mean_only_working_nodes(dummy_rec_error, dummy_cpu_alloc)
    print(ret.shape)
    print(ret)
    