import pandas as pd
import logging

data_dir = 'data/'
# load csv
df_raw = pd.read_csv(f'{data_dir}/jobs_flattened_cleaned_1.csv')

# Drop where nodes is NaN
df_raw = df_raw.dropna(subset=['nodes', 'steps-tres-requested-min-cpu-node']).reset_index(drop=True)

# Override NaN constraints values with 'no-constraints'
df_raw['constraints'] = df_raw['constraints'].fillna('no-constraints')

# Drop designated columsn
to_drop = ['reservation', 'qos', 'working_directory', 'user', 'steps-nodes-range']
# all with steps-tres ... -node
to_drop += [col for col in df_raw.columns if 'steps-tres' in col and '-node' in col]
# all starting with time- or wckey
to_drop += [col for col in df_raw.columns if col.startswith('time-') or col.startswith('wckey-')]
logging.debug(f'Drop columns: {to_drop}')
df_raw = df_raw.drop(columns=to_drop)

# Combine flags into one column
df_raw['flags'] = df_raw['flags'].apply(lambda x: '-'.join(x.split(',')))

# Drop columns which have only ONE unique value
df_raw = df_raw.loc[:, df_raw.nunique() != 1]

# Drop NaN columns
df_raw = df_raw.dropna(axis=1, how='all')

# Change all ambiguous columns to string type
ambigous = df_raw.select_dtypes(include=['object']).columns
df_raw[ambigous] = df_raw[ambigous].astype(str)

# Only jobs longer than 1 minute
df_raw = df_raw[df_raw['steps-time-elapsed'] > 60*5]

# Only jobs which consumed more than 0 energy
df_raw = df_raw[df_raw['steps-tres-consumed-total-energy'] > 0]

df_raw = df_raw.reset_index(drop=True)

redundant_columns = [
    'steps-tasks-count', 
    'steps-nodes-count', 
    
    'steps-tres-requested-max-mem', 
    'steps-tres-requested-min-mem', 
    'steps-tres-requested-average-mem', # only 1 record different than -total-mem
    
    'steps-tres-requested-min-energy', 
    'steps-tres-requested-max-energy',
    
    'steps-tres-requested-max-fs',
    'steps-tres-requested-min-fs',
    'steps-tres-requested-total-fs',
    
    'steps-tres-requested-min-vmem',
    'steps-tres-requested-max-vmem',
    
    'steps-tres-allocated-cpu',
    'steps-tres-allocated-mem',
    'steps-tres-allocated-billing',
    'steps-tres-consumed-total-fs',
    
    'tres-requested-billing',
    'tres-requested-cpu',
    'tres-requested-mem',
    'tres-requested-node',
    ]
# Potential features to drop because they might be a noise?
noisy_features = [
    'steps-statistics-CPU-actual_frequency',
    'steps-tres-requested-average-vmem',
    'steps-tres-requested-average-fs',
]

df_raw = df_raw.drop(columns=redundant_columns + noisy_features)

print(df_raw.columns)

df_raw.to_csv(f'{data_dir}jobs_step_1_final.csv', index=False)
logging.debug(f'Saved to {data_dir}/jobs_step_1_final.csv')