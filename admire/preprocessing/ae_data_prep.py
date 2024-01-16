from datetime import datetime
import json
from typing import List, Iterator
import pandas as pd
import numpy as np
import os
import logging
import configparser
import random
from itertools import cycle, repeat, compress, chain
from tqdm import tqdm
from anomaly import Anomaly
import yaml
import multiprocessing as mp

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
config = configparser.ConfigParser()
config.read('config.ini')

with open('anomalies.yaml') as f:
    anomalies = yaml.load(f, yaml.UnsafeLoader)
    anomalies_list = [Anomaly(**a) for a in anomalies['anomalies']]  

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
    
    for filename in tqdm(filenames, desc='Reading data', total=len(filenames)):
        logger.debug(f'Reading {filename}')
        path = os.path.join(raw_data_dir, filename)
        if important_cols:
            _df = pd.read_parquet(path, columns=important_cols)
        else:
            _df = pd.read_parquet(path)
        logger.debug(f'DF info {_df.info()}')
        
        _df['date']  = pd.to_datetime(_df['date'], utc=True)
        
        df = pd.concat([df, _df], ignore_index=True)          
    
    logger.debug(f'DF info {df.info()}')
    return df.reset_index(drop=True)

def augment_df(original_df: pd.DataFrame, column: str, iterator: Iterator, augment_type: int):
    copy_df = original_df.copy()
    """
    spike_all_features_iter : iterator for populating time-series with normal power,t1,t2,cpu_spikes
    spike_only_cpu_feature_iter : iterator for populating time-series with random cpu_spikes
    We want to do the second because of random spikes in cpus due to changing user allocations, it shouldn't be an error here
    TODO: make an iterator a variable passed through function to make the code cleaner
    """

    #temp = cycle(chain(spike_all_features_iter, spike_all_features_iter_0))
    iterator_res = cycle(iterator)
    indices = list(compress(copy_df.index, iterator_res))

    match column:
        case 'cpus_alloc':
            vals = 48
            copy_df.loc[indices, 'cpus_alloc'] = vals
        case 'cpu1':
            match augment_type:
                case 1:
                    vals = random.randint(48, 51)
                    copy_df.loc[indices, 'cpu1'] = vals
                case 2:
                    vals = random.randint(28, 30)
                    copy_df.loc[indices, 'cpu1'] = vals
        case 'cpu2':
            match augment_type:
                case 1:
                    vals = random.randint(48, 51)
                    copy_df.loc[indices, 'cpu2'] = vals
                case 2:
                    vals = random.randint(28, 30)
                    copy_df.loc[indices, 'cpu2'] = vals
        case 'power':
            match augment_type:
                case 1:
                    vals = random.randint(430, 490)
                    copy_df.loc[indices, 'power'] = vals
                case 2:
                    vals = random.randint(134, 146)
                    copy_df.loc[indices, 'power'] = vals

    return copy_df

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
    logger.debug(f'Saving data of shape {df.shape} to {path}')

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

def remove_data_between_dates(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Removes data between dates (inclusive, inclusive).
    """
    start = start.date()
    end = end.date()

    logger.info(f'Removing data between {start} and {end}')
    dates = pd.to_datetime(df['date']).dt.date

    return df.loc[~((dates >= start) & (dates <= end))]

def get_data_for_hosts(df: pd.DataFrame, hosts: List[str]) -> pd.DataFrame:
    """Returns/filters datafram for specified hosts only. Hosts should be a list of strings."""
    return df[df['hostname'].isin(hosts)]

def fill_missing_data(origianl_df: pd.DataFrame, date_start: datetime, date_end: datetime, host: str, data_set: str) -> pd.DataFrame:
    """Fill places where there is no measurements for a host between two dates (inclusive)"""
    _df = pd.DataFrame()
    # Create a dataframe with all dates between start and end in Warsaw timezone
    _df['date'] = pd.date_range(start=date_start, end=date_end, freq='1min', tz=datetime.now().astimezone().tzinfo)

    # convert the 'date' column to datetime format
    origianl_df = origianl_df.copy().drop_duplicates(subset="date")
    shape_before_merge = _df.shape

    # Merge the two dataframes, so that we have fixed data range for each host
    # We perform left join, so that we have all the dates from the artificial range
    # Missing values will be filled with NaN
    _df = pd.merge(_df, origianl_df, how='left', on='date')

    logger.debug(f'How many missing values {len(_df[_df["hostname"].isna()])} out of {_df.shape[0]}')

    assert shape_before_merge[0] == _df.shape[0], "length of the artificial dataframe before and after merge should be the same"

    #TODO: make a function to perform augmentation, code became to large and redundant in places
    # augment_step = random.randint(60, 90)
    # augment_len = random.randint(15, 45)

    # spike_all_features_iter = (([True]*augment_len + [False]*augment_step)*5 + ([False]*(augment_step + augment_len))*2)
    # spike_only_cpu_feature_iter = (([False]*(augment_step + augment_len))*5 + ([True]*augment_step + [False]*augment_len)*2)

    # Fill missing values with **fill_value** which is 0 by default
    match data_set:
        case 'train':
            _df['hostname'] = host
            # _df['power'] = _df['power'].fillna(augment_df(_df, 'power', spike_all_features_iter, 1)['power'])
            # _df['cpu1'] = _df['cpu1'].fillna(augment_df(_df, 'cpu1', spike_all_features_iter, 1)['cpu1'])
            # _df['cpu2'] = _df['cpu2'].fillna(augment_df(_df, 'cpu2', spike_all_features_iter, 1)['cpu2'])
            # _df['power'] = _df['power'].fillna(augment_df(_df, 'power', spike_only_cpu_feature_iter, 2)['power'])
            # _df['cpu1'] = _df['cpu1'].fillna(augment_df(_df, 'cpu1', spike_only_cpu_feature_iter, 2)['cpu1'])
            # _df['cpu2'] = _df['cpu2'].fillna(augment_df(_df, 'cpu2', spike_only_cpu_feature_iter, 2)['cpu2'])
            _df['power'] = _df['power'].fillna(random.randint(138, 144))
            _df['cpu1'] = _df['cpu1'].fillna(random.randint(28, 30))
            _df['cpu2'] = _df['cpu2'].fillna(random.randint(28, 30))
            if include_cpu_alloc:
                # _df['cpus_alloc'] = _df['cpus_alloc'].fillna(augment_df(_df, 'cpus_alloc',spike_all_features_iter, 1)['cpus_alloc'])
                # _df['cpus_alloc'] = _df['cpus_alloc'].fillna(augment_df(_df, 'cpus_alloc',spike_only_cpu_feature_iter, 1)['cpus_alloc'])
                _df['cpus_alloc'] = _df['cpus_alloc'].fillna(0)

        case 'test':
            _df['power'] = _df['power'].fillna(random.randint(138, 144))
            _df['cpu1'] = _df['cpu1'].fillna(random.randint(28, 30))
            _df['cpu2'] = _df['cpu2'].fillna(random.randint(28, 30))
            if include_cpu_alloc:
                _df['cpus_alloc'] = _df['cpus_alloc'].fillna(0)

    return _df

def host_sort(df, hosts: List[str]) -> List[str]:
    """Sort hosts by number of measurements"""
    
    logger.debug('Estimating hosts length')
    
    hostname_counts = df['hostname'].value_counts()
    hostname_counts_dict = hostname_counts.to_dict()
    
    # take only hosts that we want to process
    hostname_counts_dict = {k: v for k, v in hostname_counts_dict.items() if k in hosts}

    sorted_res = dict(sorted(hostname_counts_dict.items(), key=lambda item: item[1], reverse=True))
    return list(sorted_res.keys())[:nodes_count]

if __name__ == '__main__':
    
    raw_data_dir = config['PREPROCESSING']['raw_data_dir']
    data_dir = config['PREPROCESSING']['data_dir']
    save_data_dir = config['PREPROCESSING']['processed_data_dir']
    hosts_blacklist = config['PREPROCESSING']['hosts_blacklist'].split(',')
    nodes_count = int(config['PREPROCESSING']['nodes_count_to_process'])
    include_cpu_alloc = bool(config['PREPROCESSING']['with_cpu_alloc'])
    
    # Flush save_data_dir
    if os.path.exists(save_data_dir):
        logger.info(f'Removing {save_data_dir}')
        os.system(f'rm -rf {save_data_dir}')
        os.makedirs(save_data_dir)
        logger.info(f'Created {save_data_dir}')
    else:
        os.makedirs(save_data_dir)
        logger.info(f'Created {save_data_dir}')
    
    important_cols = ['date', 'hostname', 'power', 'cpu1', 'cpu2']
    if include_cpu_alloc:
        important_cols.append('cpus_alloc')

    logger.info(f'Loading data from {raw_data_dir}')
    raw_df = read_data(raw_data_dir, important_cols)
    logger.info(f'Loaded data shape {raw_df.shape}')
    
    # Remove hostname from columns that we want to save
    important_cols.remove('hostname')

    # Remove blacklisted hosts
    hosts_to_take = raw_df['hostname'].unique().tolist()
    hosts_to_take = list(filter(lambda x: x not in hosts_blacklist, hosts_to_take))
    
    # Sort hosts by name and take only first nodes_count
    # This is done to make sure that we always take the same hosts
    # in test and train data, but it is possible that some hosts
    # will be missing in test data if they are not in the first nodes_count
    # Therefore at the end of the script there is a sparate check to make sure
    # that all hosts from train data are also in test data (and vice versa)
    hosts_to_take.sort()
    
    # Load config for train/test data
    test_date_range_start, test_date_range_end = config['PREPROCESSING'][f'test_date_range'].split(',')
    train_date_range_start, train_date_range_end = config['PREPROCESSING'][f'train_date_range'].split(',')
    
    test_date_range_start = datetime.strptime(test_date_range_start, r'%Y-%m-%d')
    test_date_range_start = test_date_range_start.replace(tzinfo=datetime.now().astimezone().tzinfo)
    
    test_date_range_end = datetime.strptime(test_date_range_end, r'%Y-%m-%d')
    test_date_range_end = test_date_range_end.replace(tzinfo=datetime.now().astimezone().tzinfo)
    
    train_date_range_start = datetime.strptime(train_date_range_start, r'%Y-%m-%d')
    train_date_range_start = train_date_range_start.replace(tzinfo=datetime.now().astimezone().tzinfo)
    
    train_date_range_end = datetime.strptime(train_date_range_end, r'%Y-%m-%d')
    train_date_range_end = train_date_range_end.replace(tzinfo=datetime.now().astimezone().tzinfo)
    
    LOW_DATE_LIMIT = datetime.strptime('1900-01-01', r'%Y-%m-%d').replace(tzinfo=datetime.now().astimezone().tzinfo)
    HIGH_DATE_LIMIT = datetime.strptime('2100-01-01', r'%Y-%m-%d').replace(tzinfo=datetime.now().astimezone().tzinfo)

    # generate test and train data
    for type in ['train', 'test']:
        
        df = raw_df.copy(deep=True)
        
        print(df['date'].min(), df['date'].max())
        
        
        # Remove data before start date and after end date
        if type == 'test':
            df = remove_data_between_dates(df, LOW_DATE_LIMIT, test_date_range_start) 
            df = remove_data_between_dates(df, test_date_range_end, HIGH_DATE_LIMIT) 
        elif type == 'train':
            df = remove_data_between_dates(df, LOW_DATE_LIMIT, train_date_range_start) 
            df = remove_data_between_dates(df, train_date_range_end, HIGH_DATE_LIMIT)
        else:
            raise ValueError(f'Unknown value{type}')
        
        df = df.reset_index(drop=True)
         

        # Train data should not contain test data and vice versa
        # Removing test data from train data prevents data leakage 
        if type == 'train':
            # if datetime.strptime(test_date_range_start, r'%Y-%m-%d') >= datetime.strptime(train_date_range_start, r'%Y-%m-%d') and \
            #     datetime.strptime(test_date_range_end, r'%Y-%m-%d') <= datetime.strptime(train_date_range_end, r'%Y-%m-%d'):
            if test_date_range_start >= train_date_range_start and test_date_range_end <= train_date_range_end:
                    
                df = remove_data_between_dates(df, test_date_range_start, test_date_range_end)
                df = df.reset_index(drop=True)
                logger.warning(f"Train data contains test data!!! -- removing the overlap")
                logger.debug("Removed test data from train data.")
            
            # Remove blacklisted date ranges ie periods where we know that anomalies occured
            blacklisted_ranges = config['PREPROCESSING'][f'{type}_remove_periods'].split('&')
            if blacklisted_ranges[0] != '':
                for __range in blacklisted_ranges:
                    start, end = __range.split(',')
                    start = datetime.strptime(start, r'%Y-%m-%d').replace(tzinfo=datetime.now().astimezone().tzinfo)
                    end = datetime.strptime(end, r'%Y-%m-%d').replace(tzinfo=datetime.now().astimezone().tzinfo)
                    df = remove_data_between_dates(df, start, end)     
                    df = df.reset_index(drop=True)
                    
            # Do the same, but use anomalies from anomalies.yaml
            for anomaly in anomalies_list:
                df = remove_data_between_dates(df, anomaly.date_start, anomaly.date_end)
                df = df.reset_index(drop=True)
        
        logger.debug(f'After removing data {df.shape}')

        hosts = host_sort(df, hosts_to_take)
        logger.info(f'Hosts to process {hosts}')

        # Get/filter data for hosts only
        df = get_data_for_hosts(df, hosts)

        logger.info(f'After getting data for hosts {type} set {df.shape}')
        logger.info(f'Considered date range {df["date"].min()} - {df["date"].max()} for {type} set')
        
        # This function is just a wrapper in order to enable multiprocessing
        def __fill_missing_data_for_chunk_of_hosts(process_i: int, chunk: List[str], type: str, pbar: tqdm | None = None):
            for k, host in enumerate(chunk):
                logger.debug(f'Filling missing data for {host}')
                # Fill missing data for each host
                if type == 'test':
                    _df_host = fill_missing_data(df[df['hostname'] == host], test_date_range_start, test_date_range_end, host, type)
                if type == 'train':
                    _df_host = fill_missing_data(df[df['hostname'] == host], train_date_range_start, train_date_range_end, host, type)
                
                # Save each host data to a separate file named after the host
                save_data(_df_host, f'{host}', os.path.join(save_data_dir, type), keep_columns=important_cols)
                
                if (k + 1) % 5 == 0:
                    logger.info(f'Process {process_i} processed {k + 1} out of {len(chunk)} hosts')
                    if pbar:
                        pbar.update(5)
                if k == len(chunk) - 1:
                    logger.info(f'Process {process_i} processed {k + 1} out of {len(chunk)} hosts')
                    logger.info(f'Process {process_i} finished!')
        
        # Multiprocessing 
        processes = []
        cpus_available = max(1, mp.cpu_count() - 1)
        logger.info(f'Using {cpus_available} processes')
        host_chunks = np.array_split(hosts, cpus_available)
        
        # with tqdm(total=len(hosts), desc='Filling missing data') as pbar:
        for i in range(cpus_available):
            p = mp.Process(target=__fill_missing_data_for_chunk_of_hosts, args=(i, host_chunks[i].tolist(), type))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
            
        logger.info(f'Finished filling missing data for {type} set')
