import sys
import os
import datetime

import pandas as pd
import numpy as np
import plotly.express as px

from typing import Optional
from itertools import chain
from tslearn.utils import to_time_series_dataset
from admire.visualisation.visualisation import subplots_clustering #TODO change names to be more informative and function to be more general
from admire.models.data_exploration import dbscan_clustering, kmeans_clustering, local_outlier_factor, hierarchical_clustering
sys.path.append('../../')

#TODO implement calculation of n day distance matraix based on dtw_distance, saving and loading of matrix

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
ENCODING = 'label_encoding'
N_CLUSTERS = 10
print(ROOT_DIR)

def extract_power_series(df: pd.DataFrame, date: datetime.date, only_whole: bool):
    x=[]
    df_temp = df.loc[df['date'].dt.date == date]
    df_res = df_temp
    for host in df_temp['hostname'].unique():
        df_temp = df_res[df_res['hostname'] == host]
        if only_whole:
            if len(df_temp['power']) == 144:
                res = [df_temp['power'].reset_index(drop=True).to_list(), df_temp['cpu1'].reset_index(drop=True).to_list(), 
                       df_temp['cpu2'].reset_index(drop=True).to_list(),df_temp['hostname'].iloc[0],
                       df_temp['date'].reset_index(drop=True).to_list(), df_temp['cpus_alloc'].reset_index(drop=True).to_list()]
                x.append(res)
            else: pass
        else: #TODO Czy jest sens w ogóle podejmować się clusteringu pomiarów o zmiennej długości? 
            if len(df_temp['power']) != 144 and len(df_temp['power']) > 70:
                res = [df_temp['power'].reset_index(drop=True).to_list(), df_temp['cpu1'].reset_index(drop=True).to_list(), 
                    df_temp['cpu2'].reset_index(drop=True).to_list(),df_temp['hostname'].iloc[0],
                    df_temp['date'].reset_index(drop=True).to_list(), df_temp['cpus_alloc'].reset_index(drop=True).to_list()]
                x.append(res)
            else: pass

    return x
def prep_data(df: pd.DataFrame, date_min: datetime.date, date_max: datetime.date,  
              only_whole: bool, date_chosen: Optional[datetime.date] = None) -> [pd.DataFrame, np.array]:
    '''
    df : DataFrame to operate on
    date_min : First day of measurements
    date_max : Last day of measurements
    only_whole : Take only full days of measurements 
    date_chosen : Optional argument if you want to include separate day
    (Some nodes become idle due to low cluster demand, after becoming idle the measurements stop.)
    '''
    df['date'] = pd.to_datetime(df['date'])
    dates = pd.date_range(date_min, date_max)
    x=[]
    for date in dates: #Can't add date to datetimeindex, they have to be treated separately
        power_ser = extract_power_series(df, date, only_whole=only_whole)
        x.append(power_ser)
    if date_chosen:
        x.append(extract_power_series(df, date_chosen, only_whole=only_whole)) #add specific date to the calculations
    x = list(chain.from_iterable(x)) #unnest list
    df_series = pd.DataFrame(columns=['power_series', 'cpu1_series', 'cpu2', 'hostname', 'date', 'cpus_alloc'], data=x)
    if only_whole:
        x_ = to_time_series_dataset(df_series['power_series'])
        print(f"series count: {np.shape(x_)[0]}")
    else:
        x_ = to_time_series_dataset(df_series['power_series'])
        print(f"series count: {np.shape(x_)[0]}")



    return df_series, x_

def clustering(method: str):
    '''
    For data longer than 1000 series it takes a lot of time to compute
    '''
    df = pd.read_parquet(os.path.join(ROOT_DIR, 'data', '1-22.02.2023_tempdata_trimmed.parquet'))
    

    if method=="lof":
        for i in range(0,8):
            df_series, x_ = prep_data(df, datetime.date(2023, 2, 1), datetime.date(2023, 2, 3), False, datetime.date(2023, 2, 6+i))
            model, lof_labels, clusters = local_outlier_factor(x_)
            
            #subplots_clustering(clusters, lof_labels, df_series['power_series'])
            df_series['negative_outlier_factor'] = model.negative_outlier_factor_
            df_series = df_series.sort_values('negative_outlier_factor', ascending=True)
            num_of_outliers = 20
            print(df_series.iloc[0:num_of_outliers]['negative_outlier_factor'])
            print(np.average(df_series.iloc[0:num_of_outliers]['negative_outlier_factor']))
            subplots_clustering(num_of_outliers, range(0,num_of_outliers), 
                                df_series.iloc[0:num_of_outliers]['cpus_alloc'].to_numpy()) 
            subplots_clustering(num_of_outliers, range(0,num_of_outliers), 
                                df_series.iloc[0:num_of_outliers]['power_series'].to_numpy()) 
    else:
        df_series, x_ = prep_data(df, datetime.date(2023, 2, 7), datetime.date(2023, 2, 7), False)
        if method=='kmeans':
            model, clusters = kmeans_clustering(x_, n_clusters=8)
            subplots_clustering(clusters, model.labels_, df_series['power_series'])
        elif method=="dbscan":
            model, clusters = dbscan_clustering(x_)
            subplots_clustering(clusters, model.labels_, df_series['power_series'])
        
        elif method=="hierarchical":
            Z, cluster_labels, clusters = hierarchical_clustering(x_)
            subplots_clustering(clusters, cluster_labels, df_series['power_series'])


if __name__ == "__main__":
    clustering(method = 'dbscan')