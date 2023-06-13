import json
import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Hashable
from admire.preprocessing import mapping_functions
from admire.preprocessing.data_preparation import DataPreparation

data_dir = 'data/'
STEP_TO_TAKE = 0

# activate logger to log into console like print
logging.basicConfig(level=logging.DEBUG)

def fix_steps_tres_allocated_order(row):
    '''Fix order in steps-tres-allocated where node and energy are in the wrong positions'''
    ret = row[0]['tres']['allocated'][0:2]

    if row[0]['tres']['allocated'][2]['type'] == 'energy' and row[0]['tres']['allocated'][3]['type'] == 'node':
        ret += [row[0]['tres']['allocated'][3]]
    else:
        ret += [row[0]['tres']['allocated'][2]]
    row[0]['tres']['allocated'] = ret
    return row

def fix_tres_allocated_order(row):
    '''Fix order in tres-allocated where node and energy are in the wrong positions'''
    ret = row['allocated'][0:2]

    if row['allocated'][2]['type'] == 'energy' and row['allocated'][3]['type'] == 'node':
        ret += [row['allocated'][3]]
    else:
        ret += [row['allocated'][2]]
    row['allocated'] = ret
    return row

with open(f'{data_dir}jobs.json', 'r') as f:
    data = json.load(f)
    logging.debug("Json data loaded")

df_raw = pd.DataFrame.from_dict(data['jobs'])
drop_cols = ['account', 'cluster', 'container', 'comment', 'array','association', 'derived_exit_code', 'exit_code', 'het', 'job_id', 'name', 'mcs', 'kill_request_user']
drop_rows_indices = [8679]
df_raw = df_raw.drop(columns=drop_cols)
df_raw = df_raw.drop(index=drop_rows_indices).reset_index(drop=True)

# If step is greater than 0 then we want to take jobs that have exactly the number of steps 
# e.g. if we take step 1 then we filter df to records containing two steps
if STEP_TO_TAKE != 0:
    mask = df_raw['steps'].apply(lambda x: len(x) == STEP_TO_TAKE + 1) 
    df_raw = df_raw[mask].reset_index(drop=True)


df_raw['steps'] = df_raw['steps'].apply(fix_steps_tres_allocated_order)
df_raw['tres'] = df_raw['tres'].apply(fix_tres_allocated_order)

steps_series = df_raw['steps'].apply(lambda x: x[STEP_TO_TAKE] if type(x) is list and len(x) > 0 else 'ERROR')
steps_mapping_list = mapping_functions.get_mapping_for_dict(steps_series.iloc[0], 'steps')
logging.debug('Mapping for steps done')
steps_df = mapping_functions.flatten_series_by_mapping(steps_series, steps_mapping_list)
logging.debug('Flattening for steps done')
logging.debug(steps_df.columns.to_numpy())

tres_series = df_raw['tres']
tres_mapping_list = mapping_functions.get_mapping_for_dict(tres_series.iloc[0], 'tres')
tres_df = mapping_functions.flatten_series_by_mapping(tres_series, tres_mapping_list)

time_series = df_raw['time']
time_mapping_list = mapping_functions.get_mapping_for_dict(time_series.iloc[0], 'time')
time_df = mapping_functions.flatten_series_by_mapping(time_series, time_mapping_list)

required_series = df_raw['required']
required_mapping_lsit = mapping_functions.get_mapping_for_dict(required_series.iloc[0], 'required')
required_df = mapping_functions.flatten_series_by_mapping(required_series, required_mapping_lsit)

state_series = df_raw['state']
state_mapping_List = mapping_functions.get_mapping_for_dict(state_series.iloc[0], 'state')
state_df = mapping_functions.flatten_series_by_mapping(state_series, state_mapping_List)

wckey_series = df_raw['wckey']
wckey_mapping_List = mapping_functions.get_mapping_for_dict(wckey_series.iloc[0], 'wckey')
wckey_df = mapping_functions.flatten_series_by_mapping(wckey_series, wckey_mapping_List)

to_be_merged = ['steps', 'tres', 'state', 'time', 'required', 'wckey']

# Drop columns that were flattened
df_flat = df_raw.drop(columns=to_be_merged)

# Merge all dataframes
df_flat = pd.concat([df_flat, required_df, state_df, time_df, wckey_df, tres_df, steps_df], axis='columns')

assert df_flat.shape[0] == df_raw.shape[0], 'Shapes of DataFrame before and after merge should be the same!'


# # Drop columns which have only one unique value and are redundant
# cols_number = df_flat.shape[1]
# mask = []
# for col in df_flat.columns:
#     if isinstance(df_flat[col][0], list) or isinstance(df_flat[col][0], dict): #unhashable
#         mask.append(True)
#     elif len(df_flat[col].unique()) > 1:
#         mask.append(True)
#     else:
#         mask.append(False)
        

# cols_number_after = df_flat.shape[1]
# logging.debug(f'Number of columns before and after dropping redundant columns: {cols_number} -> {cols_number_after}')

# # Drop rows with missing values
# rows_numbr = df_flat.shape[0]
# df_flat = df_flat.dropna().reset_index(drop=True)
# rows_after = df_flat.shape[0]
# logging.debug(f'Number of rows before and after dropping rows with missing values: {rows_numbr} -> {rows_after}')

# befor = df_flat.shape[0]
# before_cols = df_flat.shape[1]
# df_flat = DataPreparation.clean_dataset(df_flat).reset_index(drop=True)
# after = df_flat.shape[0]
# after_cols = df_flat.shape[1]
# logging.debug(f'Number of rows before and after cleaning dataset: {befor} -> {after}')
# logging.debug(f'Number of columns before and after cleaning dataset: {before_cols} -> {after_cols}')

# -- IF STEP 1 --
# Some rows in step 1 have in column 'steps-tres-allocated-#2-type' both energy (11.7k) and node(~100). We will discard rows without energy type.
# For higher steps we do not checked
if STEP_TO_TAKE == 1:
    mask_inconsistent_jobs_on_step1 = df_flat['steps-tres-allocated-#2-type'] == 'energy'
    df_flat = df_flat[mask_inconsistent_jobs_on_step1]

# Merge tres columns to remove redundancy
df_flat_merged = mapping_functions.merge_all_tres_possible(df_flat)   
df_flat_merged = mapping_functions.remove_index_element_from_column_names(df_flat_merged)
logging.debug(df_flat_merged.columns.to_numpy())

if STEP_TO_TAKE > 0:
    df_flat.to_csv(f'{data_dir}jobs_flattened_{STEP_TO_TAKE}.csv', index=False)
    logging.info(f'Saved {data_dir}jobs_flattened_{STEP_TO_TAKE}.csv')

    df_flat_merged.to_csv(f'{data_dir}jobs_flattened_cleaned_{STEP_TO_TAKE}.csv', index=False)
    logging.info(f'Saved {data_dir}jobs_flattened_cleaned_{STEP_TO_TAKE}.csv')
else:
    df_flat.to_csv(f'{data_dir}jobs_flattened.csv', index=False)
    logging.info(f'Saved {data_dir}jobs_flattened.csv')

    df_flat_merged.to_csv(f'{data_dir}jobs_flattened_cleaned.csv', index=False)
    logging.info(f'Saved {data_dir}jobs_flattened_cleaned.csv')
