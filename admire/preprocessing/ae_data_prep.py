from typing import List
import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def read_data(filenames: List[str], data_dir: str = 'data/', important_cols: List[str] = None):
    """Reads data from parquet files, concatenates and returns a pandas dataframe"""
    df = pd.DataFrame()

    for filename in filenames:
        logger.debug(f'Reading {filename}')
        path = os.path.join(data_dir, filename)
        if important_cols:
            df = pd.concat([df, pd.read_parquet(path)[important_cols]])
        else:
            df = pd.concat([df, pd.read_parquet(path)])
    
    logger.debug(f'DF info {df.info()}')

    
    # Make sure that date is not more precise than minute
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['date'] = df['date'].dt.round('min')

    return df

def save_data(df: pd.DataFrame, filename: str, data_dir: str = 'data/processed/', keep_columns: List[str] = None):
    """Saves data to parquet file"""
    path = os.path.join(data_dir, filename)
    logger.debug(f'Saving data to {path}')

    if keep_columns:
        df[keep_columns].reset_index(drop=True).to_pickle(path)
    df.reset_index(drop=True).to_pickle(path)


def remove_data_between_dates(df: pd.DataFrame, start: str, end: str):
    """Removes data between dates (inclusive, exclusive). Format of dates is 'YYYY-MM-DD'"""
    return df[~((df['date'] >= start) & (df['date'] <= end))]


def get_data_for_hosts(df: pd.DataFrame, hosts: List[str]):
    """Returns data for hosts"""
    return df[df['hostname'].isin(hosts)]

def fill_missing_data(df: pd.DataFrame, date_start: str, date_end: str, host: str, fill_value: int = 0):
    """Fill places where there is no measurements for a host between two dates (inclusive)"""
    _df = pd.DataFrame()
    # Create a dataframe with all dates between start and end in UTC+1 timezone
    _df['date'] = pd.date_range(start=date_start, end=date_end, freq='1min', tz='Europe/Warsaw')

    # convert the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'], utc=True).astype(np.int64)
    _df['date'] = pd.to_datetime(_df['date'], utc=True).astype(np.int64)


    _df = pd.merge(_df, df, on='date', how='left')

    logger.debug(_df.head(3))
    logger.debug(f'How many missing values {len(_df[_df["hostname"].isna()])}')

    _df['hostname'] = host
    _df['power'] = _df['power'].fillna(fill_value)
    _df['cpu1'] = _df['cpu1'].fillna(fill_value)
    _df['cpu2'] = _df['cpu2'].fillna(fill_value)

    return _df

if __name__ == '__main__':
    data_dir = "data/"
    save_data_dir = "data/processed/"
    important_cols = ['date', 'hostname', 'power', 'cpu1', 'cpu2']
    files_to_read = ['01.2023_tempdata.parquet', 'temp_data1.parquet', 'temp_data2.parquet', 'temp_data3.parquet', 'temp_data4.parquet'][0:1]
    df = read_data(files_to_read, data_dir, important_cols)

    logger.debug(f'Before removing data {df.shape}')

    df = remove_data_between_dates(df, '2023-01-01', '2023-01-05')
    df = remove_data_between_dates(df, '2023-02-01', '2023-02-02')
    df = remove_data_between_dates(df, '2023-02-04', '2023-02-10')

    df = df.reset_index(drop=True)

    logger.debug(f'After removing data {df.shape}')

    hosts_blacklist = ['e2015'] # 2015 is known to be faulty
    hosts = [x for x in df['hostname'].unique() if x not in hosts_blacklist][0:10]

    df = get_data_for_hosts(df, hosts)

    logger.debug(f'After getting data for hosts {df.shape}')
    important_cols.remove('hostname')
    for host in hosts:
        logger.debug(f'Filling missing data for {host}')
        _df_host = fill_missing_data(df[df['hostname'] == host].copy(), '2023-01-05', '2023-01-31', host)
        save_data(_df_host, f'{host}.pickle', save_data_dir, keep_columns=important_cols)