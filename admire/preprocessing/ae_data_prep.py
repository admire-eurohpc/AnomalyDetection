import json
from typing import List
import pandas as pd
import numpy as np
import os
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def read_data(filenames: List[str], data_dir: str = 'data/', important_cols: List[str] = None) -> pd.DataFrame:
    """Reads data from parquet files, concatenates and returns a pandas dataframe"""
    df = pd.DataFrame()

    for filename in filenames:
        logger.debug(f'Reading {filename}')
        path = os.path.join(data_dir, filename)
        if important_cols:
            df = pd.concat([df, pd.read_parquet(path, columns=important_cols)], ignore_index=True)
        else:
            df = pd.concat([df, pd.read_parquet(path)], ignore_index=True)
    
    logger.debug(f'DF info {df.info()}')

    
    # Make sure that date is not more precise than minute
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['date'] = df['date'].dt.round('min')

    return df.reset_index(drop=True)

def save_data(df: pd.DataFrame, filename: str, data_dir: str = 'data/processed/', keep_columns: List[str] = None) -> None:
    """
    Saves data to parquet file.

    `df`: Dataframe to save
    `filename`: Name of the file to save to
    `data_dir`: Directory to save to. Default is 'data/processed/'
    `keep_columns`: If specified, only these columns will be saved
    """
    if 'parquet' not in filename:
        filename += '.parquet'
        
    if not os.path.exists(data_dir):
        logger.info(f'Creating directory {data_dir}')
        os.makedirs(data_dir)

    path = os.path.join(data_dir, filename)
    logger.info(f'Saving data of shape {df.shape} to {path}')

    # If keep_columns is specified, save only those columns
    if keep_columns:
        df[keep_columns].reset_index(drop=True).to_parquet(path)
    else:
        df.reset_index(drop=True).to_parquet(path)
        
    # Save metadata 
    # TODO: Come up with a better way to save metadata
    # as this is not very efficent and will take a lot of space
    # maybe one metadata file per whole dataset?
    # metadata = {
    #     'filename': filename,
    #     'shape': df.shape,
    #     'columns': df.columns.tolist(),
    #     'keep_columns': keep_columns,
    #     'date_index': df['date'].to_list()
    # }
    # metadata_path = os.path.join(data_dir, filename.replace('.parquet', '.json'))
    # with open(metadata_path, 'w') as f:
    #     json.dump(metadata, f)

def remove_data_between_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Removes data between dates (inclusive, exclusive). Format of dates is 'YYYY-MM-DD'.

    For example if we want to remove data between 2020-01-01 and 2020-01-05, we would call:
    `remove_data_between_dates(df, '2020-01-02', '2020-01-05')`, and this way there would be 
    data up to 2020-01-01 23:59:59 and after 2020-01-05 00:00:00

    """
    start = pd.to_datetime(start, format=r'%Y-%m-%d %H:%M:%S', utc=True)
    end = pd.to_datetime(end, format=r'%Y-%m-%d %H:%M:%S', utc=True)
    df_time = pd.to_datetime(df['date'], utc=True)
    return df[~((df_time >= start) & (df_time < end))]

def get_data_for_hosts(df: pd.DataFrame, hosts: List[str]) -> pd.DataFrame:
    """Returns data for hosts"""
    return df[df['hostname'].isin(hosts)]

def fill_missing_data(origianl_df: pd.DataFrame, date_start: str, date_end: str, host: str, fill_value: int = 0) -> pd.DataFrame:
    """Fill places where there is no measurements for a host between two dates (inclusive)"""
    _df = pd.DataFrame()
    # Create a dataframe with all dates between start and end in UTC+1 timezone
    _df['date'] = pd.date_range(start=date_start, end=date_end, freq='1min', tz='Europe/Warsaw')

    # convert the 'date' column to datetime format
    origianl_df = origianl_df.copy().drop_duplicates(subset="date")
    origianl_df['date'] = pd.to_datetime(origianl_df['date'], utc=True).astype(np.int64)
    _df['date'] = pd.to_datetime(_df['date'], utc=True).astype(np.int64)

    shape_before_merge = _df.shape

    # Merge the two dataframes, so that we have fixed data range for each host
    # We perform left join, so that we have all the dates from the artificial range
    # Missing values will be filled with NaN
    _df = pd.merge(_df, origianl_df, how='left', left_on='date', right_on='date')

    logger.debug(f'How many missing values {len(_df[_df["hostname"].isna()])}')

    assert shape_before_merge[0] == _df.shape[0], "length of the artificial dataframe before and after merge should be the same"

    # Fill missing values with **fill_value** which is 0 by default
    _df['hostname'] = host
    _df['power'] = _df['power'].fillna(fill_value)
    _df['cpu1'] = _df['cpu1'].fillna(fill_value)
    _df['cpu2'] = _df['cpu2'].fillna(fill_value)

    return _df

if __name__ == '__main__':
    data_dir = "data/"
    save_data_dir = f"data/processed/"
    important_cols = ['date', 'hostname', 'power', 'cpu1', 'cpu2']
    files_to_read = ['01.2023_tempdata.parquet', 'temp_data1.parquet', 'temp_data2.parquet', 'temp_data3.parquet', 'temp_data4.parquet'] # Only February data
    
    raw_df = read_data(files_to_read, data_dir, important_cols)
    logger.debug(f'Before removing data {raw_df.shape}')
    important_cols.remove('hostname')
    
    hosts_blacklist = ['e2015'] # 2015 is known to be faulty

    TAKE_NODES = 50
    hosts_to_take = raw_df['hostname'].unique().tolist()
    for host in hosts_blacklist:
        hosts_to_take.remove(host)
    
    hosts_to_take.sort()
    
    hosts = hosts_to_take[:TAKE_NODES]
    logging.info(f'Hosts to process {hosts}')
    
    for type in ['train', 'test']:
        
        df = raw_df.copy()
        
        if type == 'test':
            df = remove_data_between_dates(df, '2000-01-01 00:00:00', '2023-02-03') # Remove data before 2023-01-03
            df = remove_data_between_dates(df, '2023-02-13 00:00:00', '2030-05-10') # Remove data after 2023-02-13
            df = df.reset_index(drop=True)
            
            date_range_start = '2023-02-03'
            date_range_end = '2023-02-13'
              
        elif type == 'train':
            df = remove_data_between_dates(df, '2000-01-01 00:00:00', '2023-01-05 12:50:00')  # Remove data before 2023-01-05 12:50:00
            df = remove_data_between_dates(df, '2023-02-01', '2023-02-02') 
            df = remove_data_between_dates(df, '2023-02-04', '2023-02-10') # big anomaly time
            df = remove_data_between_dates(df, '2023-02-13 00:00:00', '2030-05-10') # Remove data after 2023-02-13
            
            date_range_start = '2023-01-05 12:50:00'
            date_range_end = '2023-01-31 23:59:59'

        logger.debug(f'After removing data {df.shape}')

        df = get_data_for_hosts(df, hosts)

        logger.info(f'After getting data for hosts {df.shape}')
        logger.info(f'Considered date range {df["date"].min()} - {df["date"].max()}')

        for host in tqdm(hosts, desc='Filling missing data'):
            logger.debug(f'Filling missing data for {host}')
            _df_host = fill_missing_data(df[df['hostname'] == host].copy(), date_range_start, date_range_end, host)
            save_data(_df_host, f'{host}', os.path.join(save_data_dir, type), keep_columns=important_cols)
            
    
    #TODO!: IMPORTANT - make sure that we take the same hosts for train and test
    # Remove hosts that are in train but not in test and vice versa because we need consistency
    _, _, train_host_names = next(os.walk(os.path.join(save_data_dir, 'train')))
    _, _, test_host_names = next(os.walk(os.path.join(save_data_dir, 'test')))
    
    if set(train_host_names) != set(test_host_names):
        for diff in set(train_host_names) - set(test_host_names):
            logger.error(f'{diff} is in train but not in test. Removing it')
            os.remove(os.path.join(save_data_dir, 'train', diff))
        for diff in set(test_host_names) - set(train_host_names):
            logger.error(f'{diff} is in test but not in train. Removing it')
            os.remove(os.path.join(save_data_dir, 'test', diff))
            