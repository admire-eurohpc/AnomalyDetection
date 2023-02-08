import sys

import pandas as pd
import numpy as np

sys.path.append('../../')
import os
from admire.models.data_exploration import _logistic_regression
from admire.preprocessing.data_preparation import prep_data, create_node_list

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

print(ROOT_DIR)

def main():
    df, df_raw = prep_data(os.path.join(ROOT_DIR, 'data', 'jobs_cleaned_redundant.csv'),
                   ['steps-time-user-seconds', 'steps-tres-requested-max-cpu',
                    'steps-tres-requested-max-mem', 'steps-tres-requested-max-fs', 'steps-tres-consumed-max-energy',
                    'steps-tres-consumed-max-fs'])

    node_list = create_node_list(df_raw['nodes']).reshape((12823, 1))
    print(np.shape(node_list), np.shape(df))

    data = np.concatenate((df, node_list), axis=1)
    print(np.shape(data))
    #Logistic_regression_algorithm

    #TODO wait for more data on each node
if __name__ == "__main__":
    main()