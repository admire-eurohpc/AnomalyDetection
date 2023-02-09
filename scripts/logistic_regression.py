import sys
import os
import pandas as pd
import numpy as np
from collections import Counter

from admire.models.data_exploration import _logistic_regression
from admire.preprocessing.data_preparation import DataPreparation, NumFiltering

sys.path.append('../../')

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

def main():
    df_raw = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'jobs_flattened_cleaned.csv'))

    energy_filter = NumFiltering(df_raw, 'steps-tres-consumed-max-energy')
    df_raw = energy_filter.filter_values('down', [5000])
    time_filter = NumFiltering(df_raw, 'steps-time-total-seconds')
    df_raw = time_filter.filter_values('up', [600])
    nodes_filter = NumFiltering(df_raw, 'allocation_nodes')
    df_raw = nodes_filter.filter_values('down', [2])

    counter = Counter(df_raw['nodes'])
    top_10_nodes = [node for node, _ in counter.most_common(10)]
    df_raw = df_raw[df_raw['nodes'].isin(top_10_nodes)]

    data = DataPreparation(df_raw,
                           ['allocation_nodes', 'flags', 'group', 'partition', 'priority', 'required-CPUs',
                            'steps-time-elapsed', 'steps-time-system-seconds', 'steps-time-user-seconds',
                            'steps-tres-requested-max-cpu', 'steps-tres-requested-max-mem',
                            'steps-tres-requested-max-fs', 'steps-tres-consumed-max-energy',
                            'steps-tres-consumed-max-fs']).prepare(encoding='label_encoding')


    #TODO finish logistic regression script
if __name__ == "__main__":
    main()