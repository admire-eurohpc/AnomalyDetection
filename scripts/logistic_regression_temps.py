#TODO find good use for that regression
import sys
import os
import pandas as pd
import numpy as np

from admire.models.data_exploration import _logistic_regression
from admire.preprocessing.data_preparation import MLDataPreparation, NumFiltering

sys.path.append('../../')
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
print(ROOT_DIR)
ENCODING = 'label_encoding'

def main():
    df_raw = pd.read_parquet(os.path.join(ROOT_DIR, 'data', '1-22.02.2023_tempdata.parquet'))

    print("data_prep")
    x, y = MLDataPreparation(df_raw,
                           ['cpu1', 'cpu2', 
                            'kiosk', 'rack', 'chassis', 'blade', 
                            'cpus_alloc', 'power', 
                            'hostname']).prepare(encoding=ENCODING)
    _logistic_regression(x, y, x, y, encoding=ENCODING)
if __name__ == "__main__":
    main()