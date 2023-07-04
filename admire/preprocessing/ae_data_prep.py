import json
from typing import List
import pandas as pd
import numpy as np
import os
import logging
import configparser
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
config = configparser.ConfigParser()
config.read('config.ini')

def read_data(raw_data_dir: str = 'data/raw', important_cols: List[str] = None) -> pd.DataFrame:
    """
    Reads data from parquet files, concatenates and returns a pandas dataframe.
    It will read all files in the directory specified by `raw_data_dir`.
    Passing `important_cols` will only read those columns from the files (faster).
    """
    df = pd.DataFrame()
    
    if os.getcwd() not in raw_data_dir:
        raw_data_dir = os.path.join(os.getcwd(), raw_data_dir)

    if not os.path.exists(raw_data_dir):
        logger.error(f'Raw data directory {raw_data_dir} does not exist')
        raise FileNotFoundError(f'Raw data directory {raw_data_dir} does not exist')
    
    _, _, filenames = next(os.walk(raw_data_dir))
    
    for filename in filenames:
        logger.debug(f'Reading {filename}')
        path = os.path.join(raw_data_dir, filename)
        if important_cols:
            _df = pd.read_parquet(path, columns=important_cols)
        else:
            _df = pd.read_parquet(path)
        
        _df['date']  = pd.to_datetime(_df['date'])
        
        if _df['date'].dt.tz is None:
            _df['date'] = _df['date'].dt.tz_localize('Europe/Warsaw')
        else:
            _df['date'] = _df['date'].dt.tz_convert('Europe/Warsaw')
        
        _df['date'] = _df['date'].dt.round('min')
        
        df = pd.concat([df, _df], ignore_index=True)
            
    
    logger.debug(f'DF info {df.info()}')
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
    start = pd.to_datetime(start, format=r'%Y-%m-%d', utc=False).tz_localize('Europe/Warsaw')
    end = pd.to_datetime(end, format=r'%Y-%m-%d', utc=False).tz_localize('Europe/Warsaw')

    return df.loc[~((df['date'] >= start) & (df['date'] < end))]

def get_data_for_hosts(df: pd.DataFrame, hosts: List[str]) -> pd.DataFrame:
    """Returns/filters datafram for specified hosts only. Hosts should be a list of strings."""
    return df[df['hostname'].isin(hosts)]

def fill_missing_data(origianl_df: pd.DataFrame, date_start: str, date_end: str, host: str, fill_value: int = 0) -> pd.DataFrame:
    """Fill places where there is no measurements for a host between two dates (inclusive)"""
    _df = pd.DataFrame()
    # Create a dataframe with all dates between start and end in UTC+1 timezone
    _df['date'] = pd.date_range(start=date_start, end=date_end, freq='1min', tz='Europe/Warsaw')

    # convert the 'date' column to datetime format
    origianl_df = origianl_df.copy().drop_duplicates(subset="date")
    origianl_df['date'] = pd.to_datetime(origianl_df['date'], utc=False).astype(np.int64)
    _df['date'] = pd.to_datetime(_df['date'], utc=False).astype(np.int64)

    shape_before_merge = _df.shape

    # Merge the two dataframes, so that we have fixed data range for each host
    # We perform left join, so that we have all the dates from the artificial range
    # Missing values will be filled with NaN
    _df = pd.merge(_df, origianl_df, how='left', on='date')

    logger.debug(f'How many missing values {len(_df[_df["hostname"].isna()])} out of {_df.shape[0]}')

    assert shape_before_merge[0] == _df.shape[0], "length of the artificial dataframe before and after merge should be the same"

    # Fill missing values with **fill_value** which is 0 by default
    _df['hostname'] = host
    _df['power'] = _df['power'].fillna(140)
    _df['cpu1'] = _df['cpu1'].fillna(29)
    _df['cpu2'] = _df['cpu2'].fillna(29)
    if include_cpu_alloc:
        _df['cpus_alloc'] = _df['cpus_alloc'].fillna(0)

    return _df

def host_sort(df, hosts):
    res = {}
    logger.debug('Estimating hosts length')
    for host in tqdm(hosts):
        res[host] = len(df.index[df['hostname'] == host])

    sorted_res = dict(sorted(res.items(), key=lambda item: item[1], reverse=True))
    return list(sorted_res.keys())[:TAKE_NODES]

if __name__ == '__main__':
    
    raw_data_dir = config['PREPROCESSING']['raw_data_dir']
    data_dir = config['PREPROCESSING']['data_dir']
    save_data_dir = config['PREPROCESSING']['processed_data_dir']
    hosts_blacklist = config['PREPROCESSING']['hosts_blacklist'].split(',')
    TAKE_NODES = int(config['PREPROCESSING']['nodes_count_to_process'])
    include_cpu_alloc = bool(config['PREPROCESSING']['with_cpu_alloc'])
    
    important_cols = ['date', 'hostname', 'power', 'cpu1', 'cpu2']
    if include_cpu_alloc:
        important_cols.append('cpus_alloc')

    raw_df = read_data(raw_data_dir, important_cols)
    logger.debug(f'Loaded data shape {raw_df.shape}')
    
    # Remove hostname from columns that we want to save
    important_cols.remove('hostname')

    # Remove blacklisted hosts
    hosts_to_take = raw_df['hostname'].unique().tolist()
    hosts_to_take = list(filter(lambda x: x not in hosts_blacklist, hosts_to_take))
    
    # Sort hosts by name and take only first TAKE_NODES
    # This is done to make sure that we always take the same hosts
    # in test and train data, but it is possible that some hosts
    # will be missing in test data if they are not in the first TAKE_NODES
    # Therefore at the end of the script there is a sparate check to make sure
    # that all hosts from train data are also in test data (and vice versa)
    hosts_to_take.sort()
    
    # generate test and train data
    for type in ['train', 'test']:
        
        df = raw_df.copy(deep=True)
        
        # Load config for train/test data
        date_range_start, date_range_end = config['PREPROCESSING'][f'{type}_date_range'].split(',')
        
        # Remove data before start date and after end date
        df = remove_data_between_dates(df, '2000-01-01', date_range_start) 
        df = remove_data_between_dates(df, date_range_end, '2050-01-01') 
        df = df.reset_index(drop=True)

        # Remove blacklisted date ranges
        blacklisted_ranges = config['PREPROCESSING'][f'{type}_remove_periods'].split('&')
        if blacklisted_ranges[0] != '':
            for range in blacklisted_ranges:
                start, end = range.split(',')
                df = remove_data_between_dates(df, start, end)     
                df = df.reset_index(drop=True)    

        logger.debug(f'After removing data {df.shape}')

        hosts = host_sort(df, hosts_to_take)
        logging.info(f'Hosts to process {hosts}')

        # Get/filter data for hosts only
        df = get_data_for_hosts(df, hosts)

        logger.info(f'After getting data for hosts {type} set {df.shape}')
        logger.info(f'Considered date range {df["date"].min()} - {df["date"].max()} for {type} set')

        
        for host in tqdm(hosts, desc='Filling missing data'):
            logger.debug(f'Filling missing data for {host}')
            # Fill missing data for each host
            _df_host = fill_missing_data(df[df['hostname'] == host], date_range_start, date_range_end, host)
            # Save each host data to a separate file named after the host
            save_data(_df_host, f'{host}', os.path.join(save_data_dir, type), keep_columns=important_cols)
            
    
    #TODO!: IMPORTANT - make sure that we take the same hosts for train and test
    # Remove hosts that are in train but not in test and vice versa because we need consistency
    _, _, train_host_names = next(os.walk(os.path.join(save_data_dir, 'train')))
    _, _, test_host_names = next(os.walk(os.path.join(save_data_dir, 'test')))
    
    if set(train_host_names) != set(test_host_names):
        for diff in set(train_host_names) - set(test_host_names):
            logger.error(f'{diff} is in train but not in test. Removing it')
            #os.remove(os.path.join(save_data_dir, 'train', diff))
        for diff in set(test_host_names) - set(train_host_names):
            logger.error(f'{diff} is in test but not in train. Removing it')
            #os.remove(os.path.join(save_data_dir, 'test', diff))
            