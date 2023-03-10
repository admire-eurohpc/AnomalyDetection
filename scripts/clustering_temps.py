import sys
import os
import datetime

import pandas as pd
import numpy as np

from tslearn.utils import to_time_series_dataset
from admire.visualisation.visualisation import sublots_clustering #TODO change names to be more informative and function to be more general
from admire.models.data_exploration import time_series_clustering
sys.path.append('../../')

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
ENCODING = 'label_encoding'
N_CLUSTERS = 10
print(ROOT_DIR)

def prep_data(df: pd.DataFrame, date_min: datetime.date, date_max: datetime.date, only_whole: bool) -> [list, np.array]:
    '''
    df : DataFrame to operate on
    date_min : First day of measurements
    date_max : Last day of measurements
    only_whole : Take only full days of measurements 
    (Some nodes become idle due to low cluster demand, after becoming idle the measurements stop.)
    '''
    df['date'] = pd.to_datetime(df['date'])
    dates = pd.date_range(date_min, date_max)
    x=[]
    for date in dates:
        df_temp = df.loc[df['date'].dt.date == date]
        df_res = df_temp
        for host in df_temp['hostname'].unique():
            df_temp = df_res[df_res['hostname'] == host]
            if only_whole:
                if len(df_temp['power']) == 144:
                    res = df_temp['power'].reset_index(drop=True)
                    res.name=f'{host}_{date}' #in case we want to compare distances/plot corr matrix between hosts
                    x.append(res)
                else: pass
            else: #TODO Czy jest sens w ogóle podejmować się clusteringu pomiarów o zmiennej długości? 
                res = df_temp['power'].reset_index(drop=True)
                res.name=f'{host}_{date}' #in case we want to compare distances/plot corr matrix between hosts
                x.append(res)
    if only_whole:
        x_ = np.reshape(x, (np.shape(x)[0], np.shape(x)[1], 1))
    else:
        print(np.shape(x))
        x_ = to_time_series_dataset(x)


    return x, x_

def clustering():
    '''
    For data longer than 1000 series it takes a lot of time to compute
    '''
    df = pd.read_parquet(os.path.join(ROOT_DIR, 'data', '1-22.02.2023_tempdata_trimmed.parquet'))

    x, x_ = prep_data(df, datetime.date(2023, 2, 6), datetime.date(2023, 2, 8), only_whole=True)
    
    model, clusters = time_series_clustering(x_, "dbscan")

    sublots_clustering(clusters, model.labels_, x)

if __name__ == "__main__":
    clustering()